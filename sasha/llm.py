from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, List, Optional, Tuple

from openrouter import OpenRouter
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import AgentConfig, OpenRouterConfig


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(..., description="fold|check|call|bet|raise")
    action_amount: int
    memory_summary: List[str]
    rationale: str

    @field_validator("action_type")
    @classmethod
    def _validate_action_type(cls, value: str) -> str:
        allowed = {"fold", "check", "call", "bet", "raise"}
        if value not in allowed:
            raise ValueError("action_type must be one of fold/check/call/bet/raise")
        return value

    @field_validator("memory_summary")
    @classmethod
    def _validate_memory_summary(cls, value: List[str]) -> List[str]:
        if not (2 <= len(value) <= 4):
            raise ValueError("memory_summary must have 2-4 items")
        return value

    @field_validator("action_amount")
    @classmethod
    def _validate_amount(cls, value: int) -> int:
        if value < 0:
            raise ValueError("action_amount must be >= 0")
        return value


class LLMCallResult(BaseModel):
    decision: Optional[Decision]
    response_json: Optional[str]
    latency_ms: Optional[int]
    token_usage_json: Optional[str]
    retry_count: int
    error_type: Optional[str]


class LLMClient:
    def __init__(self, openrouter_cfg: OpenRouterConfig) -> None:
        self.openrouter_cfg = openrouter_cfg

    def build_agent(self, agent_cfg: AgentConfig) -> "LLMAgent":
        client = OpenRouter(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            http_referer=self.openrouter_cfg.http_referer,
            x_title=self.openrouter_cfg.x_title,
            timeout_ms=self.openrouter_cfg.timeout_ms,
        )
        return LLMAgent(agent_cfg=agent_cfg, client=client)


class LLMAgent:
    def __init__(self, agent_cfg: AgentConfig, client: OpenRouter) -> None:
        self.agent_cfg = agent_cfg
        self.client = client
        self._memory_summary: List[str] = []

    def reset_hand_memory(self) -> None:
        self._memory_summary = []

    def get_memory_for_prompt(self) -> Optional[List[str]]:
        if self.agent_cfg.use_memory:
            return list(self._memory_summary)
        return None

    def update_memory(self, decision: Decision) -> None:
        if self.agent_cfg.use_memory:
            self._memory_summary = list(decision.memory_summary)

    def decide(
        self,
        *,
        messages: List[dict],
    ) -> LLMCallResult:
        logger = logging.getLogger("sasha.llm")
        start = time.perf_counter()

        try:
            response_format = _decision_response_format()
            response = self.client.chat.send(
                model=self.agent_cfg.model,
                messages=messages,
                response_format=response_format,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("OpenRouter request failed: %s", exc, exc_info=True)
            return LLMCallResult(
                decision=None,
                response_json=None,
                latency_ms=latency_ms,
                token_usage_json=None,
                retry_count=0,
                error_type=exc.__class__.__name__,
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        usage = response.usage
        token_usage_json = json.dumps(usage.model_dump()) if usage else None

        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice else None
        finish_reason = getattr(choice, "finish_reason", None) if choice else None

        raw_text, payload = _extract_content(content)
        if raw_text is None and payload is None:
            logger.warning(
                "Empty response content (finish_reason=%s, model=%s)",
                finish_reason,
                self.agent_cfg.model,
            )
            return LLMCallResult(
                decision=None,
                response_json=None,
                latency_ms=latency_ms,
                token_usage_json=token_usage_json,
                retry_count=0,
                error_type="EmptyResponse",
            )

        if payload is None:
            try:
                payload = json.loads(raw_text or "")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JSON parse failed (finish_reason=%s): %s; raw=%s",
                    finish_reason,
                    exc,
                    _truncate(raw_text),
                )
                return LLMCallResult(
                    decision=None,
                    response_json=raw_text,
                    latency_ms=latency_ms,
                    token_usage_json=token_usage_json,
                    retry_count=0,
                    error_type="JSONParseError",
                )

        try:
            decision = Decision.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Schema validation failed: %s; payload=%s",
                exc,
                _truncate(json.dumps(payload)),
            )
            return LLMCallResult(
                decision=None,
                response_json=json.dumps(payload),
                latency_ms=latency_ms,
                token_usage_json=token_usage_json,
                retry_count=0,
                error_type="SchemaValidationError",
            )

        return LLMCallResult(
            decision=decision,
            response_json=json.dumps(payload),
            latency_ms=latency_ms,
            token_usage_json=token_usage_json,
            retry_count=0,
            error_type=None,
        )


def _decision_response_format() -> dict:
    schema = Decision.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "Decision",
            "schema": schema,
            "strict": True,
        },
    }


def _extract_content(content: Any) -> Tuple[Optional[str], Optional[dict]]:
    if content is None:
        return None, None
    if isinstance(content, dict):
        return json.dumps(content), content
    if isinstance(content, str):
        text = content.strip()
        return text or None, None
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                continue
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
        joined = "".join(parts).strip()
        return joined or None, None
    return str(content), None


def _truncate(value: Any, limit: int = 800) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"
