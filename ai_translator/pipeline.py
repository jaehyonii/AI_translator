from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .models import (
    ContextPack,
    Direction,
    DirectionOption,
    ProcessedSource,
    StepResult,
    TextType,
    TextTypeOption,
    normalize_direction,
    normalize_text_type,
)
from .preprocess import detect_direction, detect_text_type, preprocess_source
from .prompts import (
    accuracy_verification_messages,
    classify_route,
    context_analysis_messages,
    draft_translation_messages,
    final_revision_messages,
    fluency_refinement_messages,
    rubric_scoring_messages,
    style_control_messages,
)
from .provider import LLMProvider


@dataclass
class PipelineConfig:
    direction: DirectionOption = "auto"
    text_type: TextTypeOption = "auto"
    include_score: bool = True


@dataclass
class PipelineResult:
    source_text: str
    processed: ProcessedSource
    context_pack: ContextPack
    route: str
    draft_translation: str
    fluent_translation: str
    styled_translation: str
    qa_report: str
    final_translation: str
    score_report: str = ""
    steps: list[StepResult] = field(default_factory=list)


class TranslationPipeline:
    def __init__(self, provider: LLMProvider, config: PipelineConfig | None = None) -> None:
        self.provider = provider
        self.config = config or PipelineConfig()

    def run(self, source_text: str) -> PipelineResult:
        processed = preprocess_source(source_text)
        direction = self._resolve_direction(processed.original_text)
        text_type = self._resolve_text_type(processed)

        steps: list[StepResult] = []

        context_messages = context_analysis_messages(
            processed.original_text,
            processed,
            direction,
            text_type,
        )
        context_output = self.provider.complete("context_analysis", context_messages)
        steps.append(StepResult("context_analysis", context_messages, context_output))
        context = parse_context_pack(
            context_output,
            default_direction=direction,
            default_text_type=text_type,
        )

        draft_messages = draft_translation_messages(processed.original_text, context)
        draft = self.provider.complete("draft_translation", draft_messages)
        steps.append(StepResult("draft_translation", draft_messages, draft))

        fluency_messages = fluency_refinement_messages(draft, context)
        fluent = self.provider.complete("fluency_refinement", fluency_messages)
        steps.append(StepResult("fluency_refinement", fluency_messages, fluent))

        style_messages = style_control_messages(fluent, context)
        styled = self.provider.complete("style_control", style_messages)
        steps.append(StepResult("style_control", style_messages, styled))

        qa_messages = accuracy_verification_messages(processed.original_text, styled, context)
        qa_report = self.provider.complete("accuracy_verification", qa_messages)
        steps.append(StepResult("accuracy_verification", qa_messages, qa_report))

        final_messages = final_revision_messages(styled, qa_report, context)
        final_translation = self.provider.complete("final_revision", final_messages)
        steps.append(StepResult("final_revision", final_messages, final_translation))

        score_report = ""
        if self.config.include_score:
            score_messages = rubric_scoring_messages(processed.original_text, final_translation, context)
            score_report = self.provider.complete("rubric_scoring", score_messages)
            steps.append(StepResult("rubric_scoring", score_messages, score_report))

        return PipelineResult(
            source_text=processed.original_text,
            processed=processed,
            context_pack=context,
            route=classify_route(context.direction, context.text_type),
            draft_translation=draft,
            fluent_translation=fluent,
            styled_translation=styled,
            qa_report=qa_report,
            final_translation=final_translation,
            score_report=score_report,
            steps=steps,
        )

    def _resolve_direction(self, source_text: str) -> Direction:
        if self.config.direction != "auto":
            return normalize_direction(self.config.direction)
        return detect_direction(source_text)

    def _resolve_text_type(self, processed: ProcessedSource) -> TextType:
        if self.config.text_type != "auto":
            return normalize_text_type(self.config.text_type)
        return detect_text_type(processed)


def parse_context_pack(
    raw: str,
    *,
    default_direction: Direction,
    default_text_type: TextType,
) -> ContextPack:
    parsed = extract_json_object(raw)
    if not isinstance(parsed, dict):
        parsed = {"summary": raw.strip()}
    return ContextPack.from_mapping(
        parsed,
        default_direction=default_direction,
        default_text_type=default_text_type,
        raw_analysis=raw,
    )


def extract_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None

