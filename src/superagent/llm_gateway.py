import http.client
import json
import math
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
import uuid

from .audit import write_json
from .budget import BudgetPending, BudgetStop
from .contracts import CANDIDATE_CONTRACT


class TransientLLMError(RuntimeError):
    """A bounded network recovery attempt failed without proving the call was free."""


def request_input_bound(payload, context_limit, mode="request_bytes"):
    """Conservative byte-level text bound, not a provider-certified token count."""
    if mode not in ("request_bytes", "full_context"):
        raise ValueError("input_reservation_mode must be request_bytes or full_context")
    if type(context_limit) is not int or context_limit <= 0:
        raise ValueError("context_tokens_upper_bound must be a positive integer")
    text_fields = {"role", "content"}
    supported = {"model", "messages", "max_tokens", "temperature", "stream", "stop",
                 "top_p", "frequency_penalty", "presence_penalty", "seed",
                 "thinking", "reasoning_effort"}
    messages = payload["messages"]
    plain_text = all(set(m) <= text_fields and isinstance(m.get("content"), str)
                     and m.get("role") in ("system", "user", "assistant") for m in messages)
    thinking = payload.get("thinking", {"type": "disabled"})
    known_thinking = (isinstance(thinking, dict) and set(thinking) == {"type"}
                      and thinking["type"] in ("enabled", "disabled"))
    if mode == "full_context" or not plain_text or not known_thinking or not set(payload) <= supported:
        return context_limit, {"method": "full_context", "input_tokens_upper": context_limit}
    # Byte-level tokenizers cannot exceed one token per UTF-8 byte of text.
    # Count the whole envelope as well, plus allowance for hidden chat framing.
    size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    overhead = 4096 + 256 * len(messages)
    upper = min(context_limit, size + overhead)
    return upper, {"method": "request_bytes_v1", "payload_utf8_bytes": size,
                   "template_allowance_tokens": overhead, "input_tokens_upper": upper}


class Gateway:
    def __init__(self, policy, ledger, audit, max_output_tokens=8192, request_timeout_seconds=240,
                 network_retry_attempts=0, network_retry_backoff_seconds=0,
                 network_cooldown_seconds=0, input_reservation_mode="request_bytes"):
        self.policy, self.ledger, self.audit = policy, ledger, audit
        self.max_output_tokens = max_output_tokens
        if input_reservation_mode not in ("request_bytes", "full_context"):
            raise ValueError("input_reservation_mode must be request_bytes or full_context")
        self.input_reservation_mode = input_reservation_mode
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if (type(request_timeout_seconds) not in (int, float) or not math.isfinite(request_timeout_seconds)
                or request_timeout_seconds <= 0):
            raise ValueError("request_timeout_seconds must be positive and finite")
        self.request_timeout_seconds = request_timeout_seconds
        if type(network_retry_attempts) is not int or network_retry_attempts < 0:
            raise ValueError("network_retry_attempts must be a non-negative integer")
        if (type(network_retry_backoff_seconds) not in (int, float)
                or not math.isfinite(network_retry_backoff_seconds)
                or network_retry_backoff_seconds < 0):
            raise ValueError("network_retry_backoff_seconds must be non-negative and finite")
        self.network_retry_attempts = network_retry_attempts
        self.network_retry_backoff_seconds = network_retry_backoff_seconds
        if (type(network_cooldown_seconds) not in (int, float)
                or not math.isfinite(network_cooldown_seconds) or network_cooldown_seconds < 0):
            raise ValueError("network_cooldown_seconds must be non-negative and finite")
        self.network_cooldown_seconds = network_cooldown_seconds
        self.stopped = threading.Event()
        self.stop_reason = None
        self.deadline = None
        self._condition = threading.Condition()
        self._cooldown_until = None
        self._probe_active = False
        self._failure_epoch = 0
        if not policy.get("verified"):
            raise ValueError("Model identity and historical price policy not verified; paid calls disabled")
        if policy["release_date"] > policy["cutoff_date"] or not policy["official_family"].startswith(("DeepSeek", "Qwen")):
            raise ValueError("Model outside course policy")
        self.key = os.environ.get("SUPERAGENT_API_KEY", "")
        if not self.key:
            raise ValueError("Export SUPERAGENT_API_KEY in trusted supervisor")
        self.url = os.environ.get("SUPERAGENT_BASE_URL", "https://aigc-api.hkust-gz.edu.cn/api/v2").rstrip("/") + "/chat/completions"

    def _halt(self, reason, message):
        with self._condition:
            if self.stop_reason is None:
                self.stop_reason = reason
                self.audit.event("dispatch_stopped", reason=reason)
            self.stopped.set()
            self._condition.notify_all()
        raise BudgetStop(message, self.stop_reason)

    def _check_running(self):
        if self.stopped.is_set():
            raise BudgetStop("Search stopped; no further dispatch", self.stop_reason or "budget_or_uncertain_usage")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self._halt("time_limit", "Search deadline reached")

    def _budget(self, input_limit, output_limit, *, reserve=False):
        waiting = False
        while True:
            self._check_running()
            try:
                operation = self.ledger.reserve if reserve else self.ledger.check_dispatch
                result = operation(input_limit, output_limit)
                if waiting:
                    self.audit.event("budget_resumed")
                return result
            except BudgetPending:
                if not waiting:
                    self.audit.event("budget_waiting", input_tokens_upper=input_limit,
                                     output_tokens_upper=output_limit)
                    waiting = True
                self.stopped.wait(timeout=0.25)
            except BudgetStop as error:
                self._halt(error.reason, str(error))

    def _admit(self):
        # After a transport failure, only one new request probes the service.
        # Older in-flight successes cannot close a newer circuit opening.
        with self._condition:
            while True:
                self._check_running()
                if self._cooldown_until is None:
                    return False, self._failure_epoch
                remaining = self._cooldown_until - time.monotonic()
                if remaining <= 0 and not self._probe_active:
                    self._probe_active = True
                    self.audit.event("network_probe", failure_epoch=self._failure_epoch)
                    return True, self._failure_epoch
                self._condition.wait(timeout=min(1, remaining) if remaining > 0 else 1)

    def _healthy(self, probe, epoch):
        with self._condition:
            if probe:
                self._probe_active = False
                if epoch == self._failure_epoch:
                    self._cooldown_until = None
                    self.audit.event("network_recovered", failure_epoch=epoch)
            self._condition.notify_all()

    def _unhealthy(self, probe):
        with self._condition:
            self._failure_epoch += 1
            self._cooldown_until = time.monotonic() + self.network_cooldown_seconds
            if probe:
                self._probe_active = False
            self.audit.event("network_cooldown", seconds=self.network_cooldown_seconds,
                             failure_epoch=self._failure_epoch)
            self._condition.notify_all()

    def _backoff(self, attempt):
        until = time.monotonic() + self.network_retry_backoff_seconds * (2 ** attempt)
        while True:
            self._check_running()
            remaining = until - time.monotonic()
            if remaining <= 0:
                return
            self.stopped.wait(timeout=min(1, remaining))

    def call(self, messages, function=None, temperature=None, max_tokens=None, json_schema=None, stop_tokens=None,
             provider_options=None):
        self._check_running()
        messages = [{"role": "system", "content": CANDIDATE_CONTRACT}] + messages
        schema = function.json_schema if function else json_schema
        if schema:
            messages.append({"role": "user", "content": "Return only a JSON object satisfying this schema:\n" + json.dumps(schema)})
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
            raise ValueError("max_tokens must be a positive integer")
        output_limit = min(max_tokens or self.max_output_tokens, self.max_output_tokens)
        payload = {"model": self.policy["model"], "messages": messages, "max_tokens": output_limit,
                   "temperature": 1 if temperature is None else temperature, "stream": False}
        if {"model", "messages", "max_tokens", "stream"}.intersection(provider_options or {}):
            raise ValueError("Provider options cannot override audited request fields")
        payload.update(provider_options or {})
        if stop_tokens:
            payload["stop"] = stop_tokens
        input_limit, reservation = request_input_bound(
            payload, self.policy["context_tokens_upper_bound"], self.input_reservation_mode)
        reservation["output_tokens_upper"] = output_limit
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                         headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        logical_id = uuid.uuid4().hex
        previous_call = None
        for attempt in range(self.network_retry_attempts + 1):
            self._budget(input_limit, output_limit)
            probe, epoch = self._admit()
            call_id = self._budget(input_limit, output_limit, reserve=True)
            try:
                self.audit.event("llm_reserved", call_id=call_id, model=payload["model"],
                                 logical_id=logical_id, attempt=attempt + 1, retry_of=previous_call,
                                 reservation=reservation)
                write_json(self.audit.root / "calls" / f"{call_id}.request.json", payload)
            except OSError:
                self.ledger.unknown(call_id)
                self._halt("audit_error", "Cannot persist request audit; dispatch refused")
            # Only transport/envelope decoding belongs in the uncertain-usage handler.
            # Parsing an already settled model's content must never relabel its cost.
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                    result = json.load(response)
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException,
                    json.JSONDecodeError, UnicodeDecodeError) as error:
                self.ledger.unknown(call_id)
                status = error.code if isinstance(error, urllib.error.HTTPError) else None
                certificate_error = (isinstance(error, ssl.SSLCertVerificationError)
                                     or isinstance(getattr(error, "reason", None), ssl.SSLCertVerificationError))
                retryable = not certificate_error and (status is None or status in (408, 425, 429) or 500 <= status < 600)
                if isinstance(error, urllib.error.HTTPError):
                    error.close()
                self.audit.event("llm_uncertain", call_id=call_id, error_type=type(error).__name__,
                                 http_status=status, retryable=retryable, logical_id=logical_id,
                                 attempt=attempt + 1)
                if not retryable:
                    self._halt("provider_error", "Non-retryable provider or certificate error; reservation retained")
                if self.ledger.max_unknown_calls == 0:
                    self._halt("uncertain_usage", "Strict mode: request failed; reservation retained")
                self._unhealthy(probe)
                self._budget(input_limit, output_limit)
                if attempt >= self.network_retry_attempts:
                    raise TransientLLMError("Request failed; reservation retained after bounded recovery") from None
                previous_call = call_id
                self.audit.event("llm_retry_scheduled", logical_id=logical_id, retry_of=call_id,
                                 next_attempt=attempt + 2,
                                 backoff_seconds=self.network_retry_backoff_seconds * (2 ** attempt))
                self._backoff(attempt)
                continue
            try:
                write_json(self.audit.root / "calls" / f"{call_id}.response.json", result)
            except OSError:
                self.ledger.unknown(call_id)
                self._halt("audit_error", "Cannot persist provider response; reservation retained")
            usage = result.get("usage") if isinstance(result, dict) else None
            if (not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int
                    or type(usage.get("completion_tokens")) is not int
                    or type(usage.get("total_tokens")) is not int
                    or min(usage["prompt_tokens"], usage["completion_tokens"]) < 0
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
                self.ledger.unknown(call_id)
                self._halt("invalid_usage", "Missing/inconsistent usage; reservation retained")
            try:
                self.ledger.settle(call_id, usage["prompt_tokens"], usage["completion_tokens"], usage)
            except BudgetStop as error:
                self._halt(error.reason, str(error))
            returned_model = result.get("model", "")
            if not isinstance(returned_model, str) or not any(
                    name.lower() in returned_model.lower()
                    for name in self.policy.get("allowed_response_models", [self.policy["official_family"]])):
                self._halt("model_policy_error", "Provider response model outside documented alias mapping")
            self.audit.event("llm_settled", call_id=call_id, usage=usage, response_model=returned_model)
            if usage["prompt_tokens"] > input_limit or usage["completion_tokens"] > output_limit:
                self._halt("reservation_exceeded", "Provider usage exceeded the request token bound; reconcile before continuing")
            self._healthy(probe, epoch)
            choice = result["choices"][0]
            if choice.get("finish_reason") == "length":
                raise RuntimeError("Model output truncated; charged and recorded")
            content = choice["message"].get("content") or ""
            if not content:
                raise RuntimeError("Empty model output; charged and recorded")
            if schema:
                import jsonschema
                if content.lstrip().startswith("```"):
                    content = content.strip().split("\n", 1)[1].rsplit("```", 1)[0]
                parsed = json.loads(content)
                jsonschema.validate(parsed, schema)
                return parsed if function else json.dumps(parsed)
            return content

    def generate(self, prompt, cfg=None, **kwargs):
        return self.call(normalize_prompt(prompt), temperature=kwargs.get("temperature"),
                         max_tokens=kwargs.get("max_tokens"), json_schema=kwargs.get("json_schema"),
                         stop_tokens=kwargs.get("stop_tokens"), provider_options=kwargs.get("provider_options"))

    def query(self, system_message=None, user_message=None, func_spec=None, cfg=None, **kwargs):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": stringify(system_message)})
        if user_message:
            messages.append({"role": "user", "content": stringify(user_message)})
        return self.call(messages, func_spec, temperature=kwargs.get("temperature"), max_tokens=kwargs.get("max_tokens"),
                         provider_options=kwargs.get("provider_options"))


def stringify(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def normalize_prompt(prompt):
    if isinstance(prompt, list) and all(isinstance(v, dict) and "role" in v for v in prompt):
        return prompt
    if isinstance(prompt, dict) and "messages" in prompt:
        return prompt["messages"]
    if isinstance(prompt, dict) and any(k in prompt for k in ("system", "user", "assistant")):
        return [{"role": role, "content": stringify(prompt[role])} for role in ("system", "user", "assistant") if prompt.get(role)]
    return [{"role": "user", "content": stringify(prompt)}]
