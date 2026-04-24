from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .models import LLMMessage


class LLMProvider(Protocol):
    def complete(self, step_name: str, messages: list[LLMMessage]) -> str:
        """Return a model completion for the given step."""


@dataclass
class DryRunProvider:
    """Provider that returns deterministic placeholders and never calls a network API."""

    def complete(self, step_name: str, messages: list[LLMMessage]) -> str:
        if step_name == "context_analysis":
            return json.dumps(
                {
                    "direction": "",
                    "text_type": "",
                    "summary": "DRY RUN: 실제 LLM 호출 없이 프롬프트 생성만 확인했습니다.",
                    "characters": [],
                    "plot_flow": [],
                    "style": {
                        "register": "",
                        "voice": "",
                        "sentence_length": "",
                    },
                    "key_terms": [],
                    "risk_points": [],
                },
                ensure_ascii=False,
            )
        descriptions = {
            "draft_translation": "실제 LLM 호출 시 정확성 중심 초벌 번역이 생성됩니다.",
            "fluency_refinement": "실제 LLM 호출 시 유창성 개선본이 생성됩니다.",
            "style_control": "실제 LLM 호출 시 문체 보정본이 생성됩니다.",
            "accuracy_verification": "실제 LLM 호출 시 원문-번역문 정확성 검수 결과가 생성됩니다.",
            "final_revision": "실제 LLM 호출 시 최종 제출용 번역문이 생성됩니다.",
            "rubric_scoring": "실제 LLM 호출 시 유창성/문체성/정확성 기준 점검 결과가 생성됩니다.",
        }
        return f"[DRY RUN: {step_name}]\n{descriptions.get(step_name, '프롬프트 생성만 확인했습니다.')}"


@dataclass
class OpenAICompatibleProvider:
    """Minimal OpenAI-compatible chat-completions provider using only stdlib HTTP."""

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1/chat/completions"
    temperature: float = 0.2
    timeout_seconds: int = 120

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = "LLM_API_KEY",
        model_env: str = "LLM_MODEL",
        base_url_env: str = "LLM_BASE_URL",
        temperature: float = 0.2,
    ) -> "OpenAICompatibleProvider":
        api_key = os.getenv(api_key_env, "").strip()
        model = os.getenv(model_env, "").strip()
        base_url = os.getenv(base_url_env, "https://api.openai.com/v1/chat/completions").strip()
        if not api_key:
            raise ValueError(f"{api_key_env} 환경변수가 필요합니다.")
        if not model:
            raise ValueError(f"{model_env} 환경변수가 필요합니다.")
        return cls(api_key=api_key, model=model, base_url=base_url, temperature=temperature)

    def complete(self, step_name: str, messages: list[LLMMessage]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": item.role, "content": item.content} for item in messages],
            "temperature": self.temperature,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ai-translator-pipeline/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API error during {step_name}: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API connection failed during {step_name}: {exc}") from exc

        parsed = json.loads(body)
        try:
            return str(parsed["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM API response during {step_name}: {body}") from exc
