from __future__ import annotations

import json

from .pipeline import PipelineResult
from .preprocess import render_preprocess_report


def render_markdown_report(result: PipelineResult, *, include_prompts: bool = False) -> str:
    parts = [
        "# LLM 번역 파이프라인 결과",
        "## 라우팅",
        f"- Route: {result.route}",
        f"- Direction: {result.context_pack.direction}",
        f"- Text type: {result.context_pack.text_type}",
        "## 전처리",
        fenced(render_preprocess_report(result.processed), "text"),
        "## Context Pack",
        fenced(json.dumps(result.context_pack.to_dict(), ensure_ascii=False, indent=2), "json"),
        "## 초벌 번역",
        result.draft_translation,
        "## 유창성 개선",
        result.fluent_translation,
        "## 문체 보정",
        result.styled_translation,
        "## 정확성 검수",
        result.qa_report,
        "## 최종 번역",
        result.final_translation,
    ]
    if result.score_report:
        parts.extend(["## 평가 기준 점검", result.score_report])
    if include_prompts:
        parts.append("## Prompt Log")
        for step in result.steps:
            parts.append(f"### {step.name}")
            for index, message in enumerate(step.messages, start=1):
                parts.append(f"#### {index}. {message.role}")
                parts.append(fenced(message.content, "text"))
            parts.append("#### output")
            parts.append(fenced(step.output, "text"))
    return "\n\n".join(parts).strip() + "\n"


def fenced(text: str, language: str = "") -> str:
    return f"```{language}\n{text}\n```"

