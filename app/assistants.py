"""Azure OpenAI powered assistants for natural language query generation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable

from openai import AzureOpenAI
from openai import OpenAIError

from .config import AzureOpenAIConfig


LOGGER = logging.getLogger(__name__)

ASSISTANT_EXTENSION_KEY = "assistant_service"


class AssistantServiceError(RuntimeError):
    """Wrap errors raised when interacting with the assistant."""


@dataclass(slots=True)
class AssistantService:
    """Thin wrapper around Azure OpenAI chat completions."""

    client: AzureOpenAI
    deployment: str
    assistant_id: str | None = None

    def _complete(self, system_prompt: str, user_prompt: str) -> tuple[str, Dict[str, Any]]:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except OpenAIError as exc:  # pragma: no cover - relies on remote service
            LOGGER.exception("Azure OpenAI request failed")
            raise AssistantServiceError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - safety net for SDK changes
            LOGGER.exception("Unexpected assistant failure")
            raise AssistantServiceError(str(exc)) from exc

        choice = response.choices[0]
        text = _message_content_to_text(choice.message)
        metadata = {
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": _serialise_usage(getattr(response, "usage", None)),
        }
        return text, metadata

    def suggest_jq(self, *, prompt: str, payload_excerpt: str) -> Dict[str, Any]:
        system = (
            "You assist analysts in writing JQ queries for JSON payload exploration. "
            "Respond with compact JSON using keys 'query' and 'explanation'. "
            "Always produce valid JQ snippets that can run inside jq.compile."
        )
        user = (
            "Context JSON (truncated):\n" + payload_excerpt + "\n\n"
            "User request: " + prompt
        )
        text, metadata = self._complete(system, user)
        parsed = _coerce_response_json(text)
        query = _first_non_empty(
            [
                parsed.get("query"),
                parsed.get("jq"),
                parsed.get("suggestion"),
            ]
        )
        explanation = _first_non_empty(
            [
                parsed.get("explanation"),
                parsed.get("rationale"),
                parsed.get("notes"),
                text.strip() if not query else "",
            ]
        )
        return {
            "query": query or "",
            "explanation": explanation or "",
            "raw": text,
            "metadata": metadata,
        }

    def suggest_rule(self, *, prompt: str, current_code: str, input_preview: str) -> Dict[str, Any]:
        system = (
            "You help generate Python transformation rules that map JSON payloads to"
            " pandas DataFrames. Reply with JSON keys 'code' and 'explanation'."
        )
        user_parts = [
            "Existing rule (if any):\n" + (current_code or "<empty>") + "\n",
        ]
        if input_preview:
            user_parts.append("Sample input preview:\n" + input_preview + "\n")
        user_parts.append("User request:\n" + prompt)
        text, metadata = self._complete(system, "\n".join(user_parts))
        parsed = _coerce_response_json(text)
        code = _first_non_empty(
            [parsed.get("code"), parsed.get("rule"), parsed.get("script")]
        )
        explanation = _first_non_empty(
            [
                parsed.get("explanation"),
                parsed.get("notes"),
                parsed.get("rationale"),
                text.strip() if not code else "",
            ]
        )
        return {
            "code": code or "",
            "explanation": explanation or "",
            "raw": text,
            "metadata": metadata,
        }

    def suggest_sql(
        self,
        *,
        prompt: str,
        table_name: str | None,
        schema_description: str,
        sample_rows: str,
    ) -> Dict[str, Any]:
        system = (
            "You write SQLite-compatible SQL to answer data questions. "
            "Return JSON with keys 'query' and 'explanation'."
        )
        context_parts = []
        if table_name:
            context_parts.append(f"Primary table: {table_name}")
        if schema_description:
            context_parts.append("Schema:\n" + schema_description)
        if sample_rows:
            context_parts.append("Sample rows:\n" + sample_rows)
        context_parts.append("User request:\n" + prompt)
        user_prompt = "\n\n".join(context_parts)
        text, metadata = self._complete(system, user_prompt)
        parsed = _coerce_response_json(text)
        query = _first_non_empty(
            [parsed.get("query"), parsed.get("sql"), parsed.get("statement")]
        )
        explanation = _first_non_empty(
            [
                parsed.get("explanation"),
                parsed.get("notes"),
                parsed.get("rationale"),
                text.strip() if not query else "",
            ]
        )
        return {
            "query": query or "",
            "explanation": explanation or "",
            "raw": text,
            "metadata": metadata,
        }


def build_assistant_service(settings: AzureOpenAIConfig) -> AssistantService:
    """Instantiate the assistant service if configuration is present."""

    client = AzureOpenAI(
        api_key=settings.api_key,
        azure_endpoint=settings.endpoint,
        api_version=settings.api_version,
    )
    return AssistantService(
        client=client,
        deployment=settings.deployment,
        assistant_id=settings.assistant_id,
    )


def _message_content_to_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "".join(parts)
    return ""


def _coerce_response_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Attempt to recover JSON code block content
        if "{" in stripped and "}" in stripped:
            start = stripped.find("{")
            end = stripped.rfind("}") + 1
            snippet = stripped[start:end]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
    return {}


def _first_non_empty(candidates: Iterable[Any]) -> str | None:
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _serialise_usage(usage: Any) -> Any:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    for attr in ("model_dump", "to_dict", "dict"):
        method = getattr(usage, attr, None)
        if callable(method):
            try:
                converted = method()
            except TypeError:
                try:
                    converted = method(usage)
                except Exception:  # pragma: no cover - defensive
                    converted = None
            except Exception:  # pragma: no cover - defensive
                converted = None
            if converted is not None:
                return converted
    return usage


__all__ = [
    "ASSISTANT_EXTENSION_KEY",
    "AssistantService",
    "AssistantServiceError",
    "build_assistant_service",
]
