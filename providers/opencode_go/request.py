"""Request builder for OpenCode Go provider."""

from typing import Any

from loguru import logger

from providers.common.message_converter import build_base_request_body
from providers.common.utils import map_effort_for_oss

OPENCODE_GO_DEFAULT_MAX_TOKENS = 81920


def build_request_body(request_data: Any) -> dict:
    """Build OpenAI-format request body from Anthropic request for OpenCode Go."""
    logger.debug(
        "OPENCODE_GO_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    body = build_base_request_body(
        request_data,
        default_max_tokens=OPENCODE_GO_DEFAULT_MAX_TOKENS,
        include_reasoning_for_openrouter=False,
    )

    extra_body: dict[str, Any] = {}
    request_extra = getattr(request_data, "extra_body", None)
    if request_extra:
        extra_body.update(request_extra)

    thinking = getattr(request_data, "thinking", None)
    thinking_enabled = (
        thinking.enabled if thinking and hasattr(thinking, "enabled") else False
    )
    if thinking_enabled:
        thinking_config: dict[str, Any] = {"enabled": True}
        effort = getattr(thinking, "effort", None) if thinking else None
        mapped_effort = map_effort_for_oss(effort)
        if mapped_effort:
            thinking_config["effort"] = mapped_effort
        extra_body.setdefault("thinking", thinking_config)

    if extra_body:
        body["extra_body"] = extra_body

    logger.debug(
        "OPENCODE_GO_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
