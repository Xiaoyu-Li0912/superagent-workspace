import json
import os
import sqlite3
import ssl
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from io import BytesIO

import numpy as np
import pandas as pd

from superagent.artifacts import freeze, verify_frozen
from superagent.audit import Audit, file_manifest
from superagent.budget import Ledger, BudgetPending, BudgetStop, cost_fen_units
from superagent.contracts import CandidateResult
from superagent.data import prepare_development
from superagent.evaluator import evaluate_predictions
from superagent.llm_gateway import Gateway, TransientLLMError, normalize_prompt, request_input_bound


class RequestBoundTests(unittest.TestCase):
    def test_text_counts_utf8_envelope_and_template_allowance(self):
        payload = {"messages": [{"role": "user", "content": "\u4f60\u597d"}], "max_tokens": 32}
        upper, evidence = request_input_bound(payload, 1_048_576)
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        self.assertEqual(upper, size + 4096 + 256)
        self.assertEqual(evidence["payload_utf8_bytes"], size)
        self.assertEqual(evidence["method"], "request_bytes_v1")
        self.assertEqual(request_input_bound(payload, 100)[0], 100)
        payload.update(thinking={"type": "enabled"}, reasoning_effort="high")
        self.assertEqual(request_input_bound(payload, 1_048_576)[1]["method"], "request_bytes_v1")

    def test_unsupported_inputs_and_explicit_legacy_mode_use_full_context(self):
        for payload in ({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
                        {"messages": [], "tools": []},
                        {"messages": [{"role": "tool", "content": "result"}]},
                        {"messages": [], "thinking": {"type": "enabled", "budget_tokens": 1000}}):
            with self.subTest(payload=payload):
                self.assertEqual(request_input_bound(payload, 10000)[1]["method"], "full_context")
        self.assertEqual(request_input_bound({"messages": []}, 10000, "full_context")[0], 10000)
        with self.assertRaises(ValueError):
            request_input_bound({"messages": []}, 10000, "typo")


class GatewayTests(unittest.TestCase):
    def recovering_gateway(self, *, cap=1, calls=24, unknown=3, retries=2, backoff=0, cooldown=0):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        environment = patch.dict(os.environ, {"SUPERAGENT_API_KEY": "offline-test"})
        environment.start()
        self.addCleanup(environment.stop)
        root = Path(directory.name)
        policy = {"verified": True, "release_date": "2026-10-17", "cutoff_date": "2026-10-21",
                  "official_family": "DeepSeek", "model": "DeepSeek-V4.1-Flash", "context_tokens_upper_bound": 10000}
        ledger = Ledger(root / "cost.sqlite", cap, 1, 4, calls, unknown)
        gateway = Gateway(policy, ledger, Audit(root), 16384, 900, retries, backoff, cooldown)
        return gateway, ledger

    def response(self, content="answer", model="DeepSeek-V4.1-Flash"):
        return BytesIO(json.dumps({"model": model,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode())

    def test_timeout_recovers_with_independent_reservations_and_audit_links(self):
        gateway, ledger = self.recovering_gateway()
        with patch("urllib.request.urlopen", side_effect=[TimeoutError(), self.response()]) as transport:
            self.assertEqual(gateway.call([]), "answer")
            self.assertEqual(transport.call_count, 2)
        cost = ledger.summary()
        self.assertEqual([c["state"] for c in cost["calls"]], ["unknown", "settled"])
        self.assertAlmostEqual(cost["unknown_reserved_rmb"], cost["calls"][0]["reserve"] / 1e9)
        self.assertLess(cost["unknown_reserved_rmb"], 0.075536)
        self.assertAlmostEqual(cost["settled_rmb"], 0.000030)
        events = [json.loads(line) for line in (gateway.audit.root / "events.jsonl").read_text().splitlines()]
        reserved = [e for e in events if e["event"] == "llm_reserved"]
        self.assertEqual(reserved[0]["logical_id"], reserved[1]["logical_id"])
        self.assertEqual(reserved[1]["retry_of"], reserved[0]["call_id"])
        self.assertFalse(gateway.stopped.is_set())
        self.assertTrue(any(e["event"] == "network_recovered" for e in events))

    def test_unknown_limit_stops_repeated_failures_without_fourth_dispatch(self):
        gateway, ledger = self.recovering_gateway()
        with patch("urllib.request.urlopen", side_effect=TimeoutError) as transport:
            with self.assertRaises(BudgetStop):
                gateway.call([])
            with self.assertRaises(BudgetStop):
                gateway.call([])
        self.assertEqual(transport.call_count, 3)
        self.assertEqual(ledger.summary()["unresolved_calls"], 3)
        self.assertEqual(gateway.stop_reason, "unknown_usage_limit")

    def test_recovery_cannot_exceed_budget_or_dispatch_limit(self):
        for options, reason in (({"cap": 0.1}, "budget_limit"), ({"calls": 1}, "call_limit")):
            with self.subTest(reason=reason):
                gateway, ledger = self.recovering_gateway(**options)
                with patch("urllib.request.urlopen", side_effect=TimeoutError) as transport:
                    with self.assertRaises(BudgetStop):
                        gateway.call([])
                self.assertEqual(transport.call_count, 1)
                self.assertEqual(gateway.stop_reason, reason)
                self.assertLessEqual(ledger.summary()["charged_or_reserved_rmb"], ledger.summary()["cap_rmb"])

    def test_exhausted_request_does_not_poison_other_search_branches(self):
        gateway, ledger = self.recovering_gateway(unknown=6, retries=1)
        with patch("urllib.request.urlopen", side_effect=TimeoutError) as transport:
            with self.assertRaises(TransientLLMError):
                gateway.call([])
            self.assertEqual(transport.call_count, 2)
        self.assertFalse(gateway.stopped.is_set())
        with patch("urllib.request.urlopen", return_value=self.response()):
            self.assertEqual(gateway.call([]), "answer")
        self.assertEqual(ledger.summary()["unresolved_calls"], 2)

    def test_authentication_error_is_fatal_and_http_status_is_sanitized(self):
        gateway, ledger = self.recovering_gateway()
        error = urllib.error.HTTPError("https://invalid", 401, "secret-response-text", {}, None)
        with patch("urllib.request.urlopen", side_effect=error) as transport:
            with self.assertRaises(BudgetStop):
                gateway.call([])
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(gateway.stop_reason, "provider_error")
        events = (gateway.audit.root / "events.jsonl").read_text()
        self.assertIn('"http_status": 401', events)
        self.assertNotIn("secret-response-text", events)
        self.assertEqual(ledger.summary()["unresolved_calls"], 1)

    def test_http_504_can_recover_without_claiming_failed_request_was_free(self):
        gateway, ledger = self.recovering_gateway()
        error = urllib.error.HTTPError("https://invalid", 504, "gateway", {}, None)
        with patch("urllib.request.urlopen", side_effect=[error, self.response()]):
            self.assertEqual(gateway.call([]), "answer")
        self.assertEqual(ledger.summary()["unresolved_calls"], 1)
        self.assertIn('"http_status": 504', (gateway.audit.root / "events.jsonl").read_text())

    def test_certificate_errors_are_fatal_whether_wrapped_or_direct(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                gateway, ledger = self.recovering_gateway()
                error = ssl.SSLCertVerificationError(1, "certificate verify failed")
                if wrapped:
                    error = urllib.error.URLError(error)
                with patch("urllib.request.urlopen", side_effect=error) as transport:
                    with self.assertRaises(BudgetStop):
                        gateway.call([])
                    transport.assert_called_once()
                self.assertEqual(gateway.stop_reason, "provider_error")
                self.assertEqual(ledger.summary()["unresolved_calls"], 1)

    def test_content_json_error_keeps_settled_usage_and_does_not_retry_transport(self):
        gateway, ledger = self.recovering_gateway()
        with patch("urllib.request.urlopen", return_value=self.response("not-json")) as transport:
            with self.assertRaises(json.JSONDecodeError):
                gateway.call([], json_schema={"type": "object"})
            transport.assert_called_once()
        self.assertEqual(ledger.summary()["unresolved_calls"], 0)
        self.assertEqual(ledger.summary()["calls"][0]["state"], "settled")
        self.assertFalse(gateway.stopped.is_set())

    def test_invalid_response_envelope_is_unknown_but_recoverable(self):
        gateway, ledger = self.recovering_gateway()
        with patch("urllib.request.urlopen", side_effect=[BytesIO(b"not-json"), self.response()]):
            self.assertEqual(gateway.call([]), "answer")
        self.assertEqual(ledger.summary()["unresolved_calls"], 1)

    def test_missing_usage_and_wrong_model_are_not_retried(self):
        gateway, ledger = self.recovering_gateway()
        with patch("urllib.request.urlopen", return_value=BytesIO(b'{"choices": []}')) as transport:
            with self.assertRaises(BudgetStop):
                gateway.call([])
            transport.assert_called_once()
        self.assertEqual(gateway.stop_reason, "invalid_usage")
        gateway, ledger = self.recovering_gateway()
        with patch("urllib.request.urlopen", return_value=self.response(model="disallowed")) as transport:
            with self.assertRaises(BudgetStop):
                gateway.call([])
            transport.assert_called_once()
        self.assertEqual(gateway.stop_reason, "model_policy_error")
        self.assertEqual(ledger.summary()["unresolved_calls"], 0)

    def test_deadline_prevents_retry_after_first_failed_dispatch(self):
        gateway, ledger = self.recovering_gateway(backoff=30)
        gateway.deadline = time.monotonic() + 60

        def timeout(*args, **kwargs):
            gateway.deadline = time.monotonic() - 1
            raise TimeoutError

        with patch("urllib.request.urlopen", side_effect=timeout) as transport:
            with self.assertRaises(BudgetStop):
                gateway.call([])
            transport.assert_called_once()
        self.assertEqual(gateway.stop_reason, "time_limit")
        self.assertEqual(ledger.summary()["unresolved_calls"], 1)

    def test_circuit_allows_only_one_probe_and_stale_success_cannot_close_it(self):
        gateway, _ = self.recovering_gateway()
        old_permit = gateway._admit()
        gateway._unhealthy(False)
        gateway._healthy(*old_permit)
        self.assertIsNotNone(gateway._cooldown_until)
        permit = gateway._admit()
        entered = threading.Event()

        def waiter():
            result = gateway._admit()
            entered.set()
            return result

        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(waiter)
            self.assertFalse(entered.wait(0.05))
            gateway._healthy(*permit)
            self.assertFalse(future.result(timeout=2)[0])
        self.assertTrue(entered.is_set())

    def test_provider_options_cannot_override_reserved_output_limit(self):
        gateway, ledger = self.recovering_gateway()
        with self.assertRaises(ValueError):
            gateway.call([], provider_options={"max_tokens": 999999})
        self.assertEqual(ledger.summary()["calls"], [])

    def test_reservation_counts_contract_schema_and_records_bounds(self):
        gateway, ledger = self.recovering_gateway(retries=0)
        gateway.policy["context_tokens_upper_bound"] = 1_048_576
        schema = {"type": "object", "description": "x" * 2000}
        with patch("urllib.request.urlopen", return_value=self.response("{}")):
            gateway.call([{"role": "user", "content": "hello"}], json_schema=schema)
        payload = json.loads(next((gateway.audit.root / "calls").glob("*.request.json")).read_text())
        upper, evidence = request_input_bound(payload, 1_048_576)
        self.assertEqual(len(payload["messages"]), 3)
        self.assertEqual(ledger.summary()["calls"][0]["reserve"], cost_fen_units(upper, 16384, 1, 4))
        events = [json.loads(line) for line in (gateway.audit.root / "events.jsonl").read_text().splitlines()]
        reserved = next(e for e in events if e["event"] == "llm_reserved")
        self.assertEqual(reserved["reservation"]["input_tokens_upper"], evidence["input_tokens_upper"])

    def test_waiting_budget_resumes_or_stops_when_inflight_becomes_unknown(self):
        for resolve_unknown in (False, True):
            with self.subTest(resolve_unknown=resolve_unknown):
                gateway, ledger = self.recovering_gateway(cap=0.1)
                blocker = ledger.reserve(80000, 0)
                waiting = threading.Event()
                original_event = gateway.audit.event

                def record(event, **fields):
                    original_event(event, **fields)
                    if event == "budget_waiting":
                        waiting.set()

                with patch.object(gateway.audit, "event", side_effect=record), \
                     patch("urllib.request.urlopen", return_value=self.response()) as transport, \
                     ThreadPoolExecutor(1) as pool:
                    future = pool.submit(gateway.call, [])
                    try:
                        self.assertTrue(waiting.wait(2))
                        transport.assert_not_called()
                    finally:
                        if resolve_unknown:
                            ledger.unknown(blocker)
                        else:
                            ledger.settle(blocker, 10, 0, {})
                    if resolve_unknown:
                        with self.assertRaises(BudgetStop):
                            future.result(timeout=3)
                        self.assertEqual(gateway.stop_reason, "budget_limit")
                    else:
                        self.assertEqual(future.result(timeout=3), "answer")
                        self.assertFalse(gateway.stopped.is_set())
                        self.assertIn('"event": "budget_resumed"', (gateway.audit.root / "events.jsonl").read_text())

    def test_budget_wait_respects_deadline_even_at_atomic_reservation(self):
        gateway, ledger = self.recovering_gateway(cap=0.1)
        ledger.reserve(80000, 0)
        gateway.deadline = time.monotonic() + 0.05
        with self.assertRaises(BudgetStop):
            gateway._budget(10000, 16384, reserve=True)
        self.assertEqual(gateway.stop_reason, "time_limit")
        self.assertEqual(len(ledger.summary()["calls"]), 1)

    def test_input_bound_violation_halts_even_if_total_cost_fits_reservation(self):
        gateway, ledger = self.recovering_gateway()
        response = {"model": "DeepSeek", "usage": {"prompt_tokens": 10001, "completion_tokens": 0,
                    "total_tokens": 10001}, "choices": [{"message": {"content": "answer"}}]}
        with patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(response).encode())):
            with self.assertRaises(BudgetStop):
                gateway.call([])
        self.assertEqual(gateway.stop_reason, "reservation_exceeded")
        self.assertAlmostEqual(ledger.summary()["settled_rmb"], 0.010001)

    def test_generation_preserves_temperature_schema_and_stops_on_unknown_usage(self):
        policy = {"verified": True, "release_date": "2026-10-17", "cutoff_date": "2026-10-21",
                  "official_family": "DeepSeek", "model": "DeepSeek-V4.1-Flash",
                  "context_tokens_upper_bound": 10000}
        response = {"model": policy["model"], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    "choices": [{"finish_reason": "stop", "message": {"content": '{"plan":"ok"}'}}]}
        schema = {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SUPERAGENT_API_KEY": "offline-test"}):
            root = Path(directory)
            ledger = Ledger(root / "cost.sqlite", 1, 1, 4)
            gateway = Gateway(policy, ledger, Audit(root))
            with patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(response).encode())) as transport:
                result = gateway.generate("plan", temperature=0.8, json_schema=schema, stop_tokens=["END"])
                self.assertEqual(json.loads(result), {"plan": "ok"})
                payload = json.loads(transport.call_args.args[0].data)
                self.assertEqual(payload["temperature"], 0.8)
                self.assertEqual(payload["stop"], ["END"])
                self.assertEqual(transport.call_args.kwargs["timeout"], 240)
            response["usage"]["total_tokens"] = 999
            with patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(response).encode())) as transport:
                with self.assertRaises(BudgetStop):
                    gateway.generate("plan")
                with self.assertRaises(BudgetStop):
                    gateway.generate("plan")
                transport.assert_called_once()
            self.assertEqual(ledger.summary()["unresolved_calls"], 1)

    def test_transport_usage_and_no_reasoning_execution(self):
        policy = {"verified": True, "release_date": "2026-10-17", "cutoff_date": "2026-10-21",
                  "official_family": "DeepSeek", "model": "DeepSeek-V4.1-Flash",
                  "context_tokens_upper_bound": 10000}
        response = {"model": policy["model"], "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}]}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SUPERAGENT_API_KEY": "offline-test"}):
            root = Path(directory)
            ledger = Ledger(root / "cost.sqlite", 1, 1, 4)
            gateway = Gateway(policy, ledger, Audit(root))
            with patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(response).encode())) as transport:
                self.assertEqual(gateway.call([{"role": "user", "content": "test"}]), "answer")
                payload = json.loads(transport.call_args.args[0].data)
                self.assertNotIn("chat_template_kwargs", payload)
                self.assertNotIn("thinking", payload)
            response["choices"][0]["message"] = {"content": None, "reasoning_content": "not executable"}
            with patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(response).encode())):
                with self.assertRaisesRegex(RuntimeError, "Empty model output"):
                    gateway.call([])
            self.assertEqual(ledger.summary()["input_tokens"], 200)
            self.assertEqual(ledger.summary()["output_tokens"], 100)
            self.assertEqual(ledger.summary()["unresolved_calls"], 0)

    def test_configured_timeout_and_output_cap_are_sent_to_provider(self):
        policy = {"verified": True, "release_date": "2026-10-17", "cutoff_date": "2026-10-21",
                  "official_family": "DeepSeek", "model": "DeepSeek-V4.1-Flash",
                  "context_tokens_upper_bound": 10000}
        response = {"model": policy["model"], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}]}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SUPERAGENT_API_KEY": "offline-test"}):
            root = Path(directory)
            ledger = Ledger(root / "cost.sqlite", 1, 1, 4)
            gateway = Gateway(policy, ledger, Audit(root), max_output_tokens=16384, request_timeout_seconds=900)
            with patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(response).encode())) as transport:
                self.assertEqual(gateway.call([{"role": "user", "content": "test"}]), "answer")
                payload = json.loads(transport.call_args.args[0].data)
                self.assertEqual(payload["max_tokens"], 16384)
                self.assertEqual(transport.call_args.kwargs["timeout"], 900)
            with patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(response).encode())) as transport:
                gateway.call([], max_tokens=2048)
                self.assertEqual(json.loads(transport.call_args.args[0].data)["max_tokens"], 2048)
            with patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(response).encode())) as transport:
                gateway.call([], max_tokens=32768)
                self.assertEqual(json.loads(transport.call_args.args[0].data)["max_tokens"], 16384)

    def test_timeout_retains_reservation_and_never_retries(self):
        policy = {"verified": True, "release_date": "2026-10-17", "cutoff_date": "2026-10-21",
                  "official_family": "DeepSeek", "model": "DeepSeek-V4.1-Flash",
                  "context_tokens_upper_bound": 10000}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SUPERAGENT_API_KEY": "offline-test"}):
            root = Path(directory)
            ledger = Ledger(root / "cost.sqlite", 1, 1, 4)
            gateway = Gateway(policy, ledger, Audit(root), max_output_tokens=16384, request_timeout_seconds=900)
            with patch("urllib.request.urlopen", side_effect=TimeoutError) as transport:
                with self.assertRaises(BudgetStop):
                    gateway.call([])
                with self.assertRaises(BudgetStop):
                    gateway.call([])
                transport.assert_called_once()
                self.assertEqual(transport.call_args.kwargs["timeout"], 900)
            self.assertEqual(ledger.summary()["unresolved_calls"], 1)
            self.assertLess(ledger.summary()["charged_or_reserved_rmb"], 0.075536)
            self.assertGreater(ledger.summary()["charged_or_reserved_rmb"], 0.065536)


class BudgetTests(unittest.TestCase):
    def test_pending_only_if_inflight_could_release_sufficient_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / "cost.sqlite", 1, 1, 4, 50, 3)
            unknown = ledger.reserve(600000, 0)
            ledger.unknown(unknown)
            ledger.reserve(300000, 0)
            with self.assertRaises(BudgetPending):
                ledger.check_dispatch(200000, 0)
            with self.assertRaises(BudgetStop) as caught:
                ledger.check_dispatch(500000, 0)
            self.assertNotIsInstance(caught.exception, BudgetPending)
            self.assertEqual(caught.exception.reason, "budget_limit")

    def test_unknown_recovery_policy_is_persistent_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cost.sqlite"
            ledger = Ledger(path, 1, 1, 4, max_unknown_calls=3)
            first = ledger.reserve(1000, 1000)
            ledger.unknown(first)
            second = Ledger(path, 1, 1, 4, max_unknown_calls=3).reserve(1000, 1000)
            self.assertNotEqual(first, second)
            with self.assertRaises(ValueError):
                Ledger(path, 1, 1, 4, max_unknown_calls=4)
            with self.assertRaises(ValueError):
                Ledger(path, 1, 1, 4)

    def test_concurrent_reservations_count_unknown_and_inflight_against_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / "cost.sqlite", 1, 1, 4, 50, 3)
            ledger.unknown(ledger.reserve(100_000, 0))

            def reserve(_):
                try:
                    return ledger.reserve(100_000, 0)
                except BudgetStop:
                    return None

            with ThreadPoolExecutor(16) as pool:
                ids = [i for i in pool.map(reserve, range(30)) if i]
            self.assertEqual(len(ids), 9)
            summary = ledger.summary()
            self.assertEqual(summary["charged_or_reserved_rmb"], 1)
            self.assertEqual(summary["unknown_reserved_rmb"], 0.1)
            self.assertEqual(summary["inflight_reserved_rmb"], 0.9)

    def test_concurrent_reservations_and_settlement(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / "cost.sqlite", 1, 1, 4, 50)
            def reserve(_):
                try:
                    return ledger.reserve(100_000, 0)
                except BudgetStop:
                    return None
            with ThreadPoolExecutor(16) as pool:
                ids = [i for i in pool.map(reserve, range(30)) if i]
            self.assertEqual(len(ids), 10)
            self.assertEqual(ledger.summary()["charged_or_reserved_rmb"], 1)
            ledger.settle(ids[0], 0, 0, {"prompt_tokens": 0, "completion_tokens": 0})
            self.assertIsNotNone(ledger.reserve(100_000, 0))

    def test_unknown_persists_and_halts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cost.sqlite"
            ledger = Ledger(path, 1, 1, 4)
            call = ledger.reserve(1000, 1000)
            with self.assertRaises(BudgetStop):
                ledger.settle(call, None, None, {})
            with self.assertRaises(BudgetStop):
                Ledger(path, 1, 1, 4).reserve(1, 1)
            self.assertEqual(ledger.summary()["unresolved_calls"], 1)

    def test_policy_immutable_and_rounding(self):
        self.assertEqual(cost_fen_units(1, 1, "1", "4"), 5000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cost.sqlite"
            Ledger(path, 1, 1, 4)
            with self.assertRaises(ValueError):
                Ledger(path, 30, 1, 4)


class DataTests(unittest.TestCase):
    def test_dev_never_needs_test_and_labels_not_exported_to_val(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("train", "val"):
                pd.DataFrame({"a": ["red", "green", "red"], "b": ["x", "y", "z"], "target": [0, 1, 0]}).to_csv(root / f"{name}.csv", index=False)
            spec = {"format": "csv", "modality": "text_pair", "train_features": "train.csv", "val_features": "val.csv",
                    "test_features": "DOES_NOT_EXIST.csv", "label_column": "target", "feature_columns": ["a", "b"]}
            run = root / "run"
            run.mkdir()
            profile = prepare_development(spec, root, run)
            self.assertEqual(profile["classes"], [0, 1])
            self.assertFalse((run / "data/val/labels.csv").exists())
            self.assertEqual(list(pd.read_csv(run / "data/val/features.csv").columns), ["a", "b"])

    def test_predictions_order_domain_and_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p.csv"
            pd.DataFrame({"row_id": [0, 1], "prediction": [1, 0]}).to_csv(path, index=False)
            self.assertEqual(evaluate_predictions(path, np.array([0, 1]), [0, 1]), 0)
            pd.DataFrame({"row_id": [1, 0], "prediction": [1, 0]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                evaluate_predictions(path, np.array([0, 1]), [0, 1])


class ArtifactTests(unittest.TestCase):
    def test_search_initialization_failure_preserves_cost_and_failure_summary(self):
        from superagent.orchestrator import run_search
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "failed-run"
            runtime = {"steps": 100, "max_calls": 400, "candidate_memory_mb": 4096,
                       "candidate_cpus": 2, "max_output_tokens": 32768,
                       "time_limit_seconds": 28800, "candidate_timeout_seconds": 600,
                       "request_timeout_seconds": 1200, "budget_rmb": 20}
            policy = {"input_rmb_per_million": 1, "output_rmb_per_million": 4}
            with patch("superagent.orchestrator.prepare_development", return_value={}), \
                 patch("superagent.orchestrator.subprocess.check_output", return_value="offline-test"), \
                 patch("superagent.orchestrator.logging.FileHandler"), \
                 patch("superagent.orchestrator.Gateway", side_effect=ValueError("missing credential")):
                with self.assertRaisesRegex(ValueError, "missing credential"):
                    run_search(root, {}, root / "unused-data", run, runtime, policy)
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["stop_reason"], "initialization_error")
            self.assertEqual(summary["cost"]["calls"], [])
            self.assertTrue((run / "cost_summary.json").exists())

    def test_prediction_uses_frozen_run_resources_with_legacy_defaults(self):
        from superagent.orchestrator import predict
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for runtime, expected in ((None, (300, 4096, 2)),
                                      ({"candidate_timeout_seconds": 600,
                                        "candidate_memory_mb": 8192, "candidate_cpus": 4}, (600, 8192, 4))):
                with self.subTest(runtime=runtime):
                    if runtime is not None:
                        (run / "runtime.json").write_text(json.dumps(runtime))
                    with patch("superagent.orchestrator.verify_frozen") as verify, \
                         patch("superagent.orchestrator.Sandbox") as sandbox:
                        output = run / ("legacy-output" if runtime is None else "configured-output")
                        predict(run, run / "features", output)
                        sandbox.assert_called_once_with(*expected)
                        sandbox.return_value.run.assert_called_once()
                        self.assertEqual(verify.call_count, 2)

    def test_frozen_hashes_detect_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            candidate = run / "candidate"
            (candidate / "model").mkdir(parents=True)
            (candidate / "model/model.txt").write_text("model")
            (candidate / "solution.py").write_text("print('ok')")
            freeze(candidate, run, {"metric": 0})
            verify_frozen(run)
            (run / "best/solution.py").write_text("modified")
            with self.assertRaises(ValueError):
                verify_frozen(run)

    def test_chat_normalization(self):
        value = normalize_prompt({"system": "s", "user": "u", "assistant": "a"})
        self.assertEqual([v["role"] for v in value], ["system", "user", "assistant"])


@unittest.skipUnless(sys.platform == "linux", "Requires Linux bubblewrap")
class SandboxTests(unittest.TestCase):
    def test_filesystem_network_secrets_and_reload(self):
        from superagent.runner import Sandbox
        code = '''import argparse, os, json, socket
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('mode'); p.add_argument('--data'); p.add_argument('--model'); p.add_argument('--output'); a=p.parse_args()
assert 'SUPERAGENT_API_KEY' not in os.environ
assert not Path('/mnt').exists() and not Path('/home').exists()
assert not Path('/data/validation_labels.npy').exists()
assert not Path('/data/y_test.npy').exists()
try:
    socket.create_connection(('1.1.1.1',443),timeout=1)
except OSError: pass
else: raise RuntimeError('Network reachable')
if a.mode=='train':
    y=pd.read_csv(Path(a.data)/'labels.csv'); (Path(a.model)/'label.json').write_text(json.dumps(int(y.label.mode()[0])))
else:
    assert not (Path(a.data)/'labels.csv').exists()
    label=json.loads((Path(a.model)/'label.json').read_text())
    x=pd.read_csv(Path(a.data)/'features.csv'); pd.DataFrame({'row_id':range(len(x)), 'prediction':[label]*len(x)}).to_csv(a.output,index=False)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solution = root / "solution.py"
            solution.write_text(code)
            for split in ("train", "val"):
                (root / split).mkdir()
                pd.DataFrame({"feature": [1, 2]}).to_csv(root / split / "features.csv", index=False)
            pd.DataFrame({"label": [1, 1]}).to_csv(root / "train/labels.csv", index=False)
            runner = Sandbox(timeout=10)
            os.environ["SUPERAGENT_API_KEY"] = "test-secret-not-a-real-key"
            try:
                runner.run(solution, root / "train", root / "model", root / "train_output", "train", root / "train.log")
                runner.run(solution, root / "val", root / "model", root / "val_output", "predict", root / "predict.log")
            finally:
                os.environ.pop("SUPERAGENT_API_KEY", None)
            self.assertEqual(evaluate_predictions(root / "val_output/predictions.csv", np.array([1, 1]), [0, 1]), 1)

if __name__ == "__main__":
    unittest.main()
