# Modified for AIAA3102 course integration; see MLEvolve/LOCAL_CHANGES.md.
import logging
from . import gemini as _gemini
from . import openai as _openai
from .gemini import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
from config import Config
logger = logging.getLogger("MLEvolve")

_transport = None


class SearchStopped(BaseException):
    """A host budget/deadline stop must bypass agent-level retry handlers."""


def set_transport(transport):
    """Optional audited host transport; native provider routing is the default."""
    global _transport
    _transport = transport


def _host_call(method, **kwargs):
    from .model_profiles import get_profile, get_thinking_extra_body, thinking_json_incompatible
    model = kwargs.get("model") or kwargs["cfg"].agent.code.model
    schema = kwargs.get("json_schema")
    if method == "generate" and schema is not None and thinking_json_incompatible(model):
        kwargs["json_schema"] = schema = None
    thinking = ((kwargs.get("func_spec") is None) if method == "query" else schema is None) or model.lower().startswith("claude")
    profile = get_profile(model, use_thinking=thinking)
    options = {key: value for key, value in profile.items()
               if key in {"temperature", "top_p", "presence_penalty", "top_k", "enable_thinking"}}
    if thinking:
        options.update(get_thinking_extra_body(model))
    kwargs["provider_options"] = options
    try:
        return getattr(_transport, method)(**kwargs)
    except Exception as error:
        if getattr(error, "stop_search", False):
            raise SearchStopped(str(error)) from error
        raise


def _provider(model: str) -> str:
    """Use Gemini backend for model names starting with 'gemini', else OpenAI-compatible (e.g. Qwen)."""
    return "gemini" if (model or "").lower().startswith("gemini") else "openai"


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    cfg:Config=None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gemini-3-pro-preview")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    if _transport is not None:
        return _host_call("query", system_message=compile_prompt_to_md(system_message) if system_message else None,
                          user_message=compile_prompt_to_md(user_message) if user_message else None,
                          model=model, temperature=temperature, max_tokens=max_tokens,
                          func_spec=func_spec, cfg=cfg, **model_kwargs)

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info("---Querying model---", extra={"verbose": True})
    system_message = compile_prompt_to_md(system_message) if system_message else None
    if system_message:
        if len(system_message) > 1000:
            logger.info(f"system: {system_message[-1000:]}", extra={"verbose": True})
        else:
            logger.info(f"system: {system_message}", extra={"verbose": True})
    user_message = compile_prompt_to_md(user_message) if user_message else None
    if user_message:
        if len(user_message) > 1000:
            logger.info(f"user: {user_message[-1000:]}", extra={"verbose": True})
        else:
            logger.info(f"user: {user_message}", extra={"verbose": True})
    if func_spec:
        logger.info(f"function spec: {func_spec.to_dict()}", extra={"verbose": True})

    provider = _provider(model)
    if provider == "openai":
        output, req_time, in_tok_count, out_tok_count, info = _openai.query(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            cfg=cfg,
            **model_kwargs,
        )
    else:
        output, req_time, in_tok_count, out_tok_count, info = _gemini.query(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            cfg=cfg,
            **model_kwargs,
        )
    logger.info("---Query complete---", extra={"verbose": True})

    return output


def generate(
    prompt,
    cfg,
    temperature=None,
    max_tokens=None,
    stop_tokens=None,
    json_schema=None,
    max_retries=20,
    retry_delay=3,
):
    """Streaming text generation. Dispatches to Gemini or OpenAI-compatible backend by cfg.agent.code.model."""
    if _transport is not None:
        return _host_call("generate", prompt=prompt, cfg=cfg, temperature=temperature,
                          max_tokens=max_tokens, stop_tokens=stop_tokens, json_schema=json_schema)
    model = getattr(cfg.agent.code, "model", "") or ""
    if _provider(model) == "openai":
        return _openai.generate(
            prompt=prompt,
            cfg=cfg,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
            json_schema=json_schema,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
    return _gemini.generate(
        prompt=prompt,
        cfg=cfg,
        temperature=temperature,
        max_tokens=max_tokens,
        stop_tokens=stop_tokens,
        json_schema=json_schema,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
