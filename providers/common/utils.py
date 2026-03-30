"""Shared utility helpers for provider request builders."""

from typing import Any


def set_if_not_none(body: dict[str, Any], key: str, value: Any) -> None:
    """Set body[key] = value only when value is not None."""
    if value is not None:
        body[key] = value


def map_effort_for_oss(effort: Any) -> str | None:
    """Map effort level for OSS models.

    OSS models don't support 'max' effort, so we map it to 'high'.
    """
    if effort is None:
        return None
    if isinstance(effort, str) and effort == "max":
        return "high"
    if hasattr(effort, "value"):
        if effort.value == "max":
            return "high"
        return effort.value
    return None
