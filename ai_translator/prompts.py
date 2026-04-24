from __future__ import annotations

import json

from .models import ContextPack, Direction, LLMMessage, ProcessedSource, TextType
from .preprocess import render_preprocess_report


ROUTE_LABELS: dict[tuple[Direction, TextType], str] = {
    ("en-ko", "dialogue"): "영한번역 - 대화체",
    ("en-ko", "narrative"): "영한번역 - 서술체",
    ("ko-en", "dialogue"): "한영번역 - 대화체",
    ("ko-en", "narrative"): "한영번역 - 서술체",
}


def classify_route(direction: Direction, text_type: TextType) -> str:
    return ROUTE_LABELS[(direction, text_type)]


def context_analysis_messages(
    source_text: str,
    processed: ProcessedSource,
    direction: Direction,
    text_type: TextType,
) -> list[LLMMessage]:
    return [
        LLMMessage(
            role="system",
            content=(
                "너는 전문 번역가의 맥락 분석 보조자다. 번역하지 말고, 번역 품질을 높이는 데 필요한 "
                "분석만 구조화한다. 출력은 반드시 JSON 객체 하나만 사용한다."
            ),
        ),
        LLMMessage(
            role="user",
            content=f"""아래 원문을 번역하기 전에 분석하라.

자동 분류 후보:
- 번역 방향: {direction}
- 글의 종류: {text_type}
- 라우트: {classify_route(direction, text_type)}

전처리 결과:
{render_preprocess_report(processed)}

출력 JSON 형식:
{{
  "direction": "en-ko 또는 ko-en",
  "text_type": "dialogue 또는 narrative",
  "summary": "전체 내용 요약",
  "characters": [
    {{
      "name": "인물명",
      "role": "화자/청자/서술 대상",
      "relationship": "관계",
      "tone": "말투와 감정"
    }}
  ],
  "plot_flow": ["사건 또는 논리 흐름 1", "사건 또는 논리 흐름 2"],
  "style": {{
    "register": "formal/casual/literary/academic 등",
    "voice": "calm/emotional/persuasive/humorous 등",
    "sentence_length": "short/medium/long"
  }},
  "key_terms": [
    {{
      "source": "원문 표현",
      "translation": "권장 번역 또는 빈 문자열",
      "note": "주의점"
    }}
  ],
  "risk_points": ["오역 가능성이 높은 표현", "직역하면 어색한 문장"]
}}

분석 항목:
1. 번역 방향
2. 글의 종류: 대화체/서술체
3. 전체 요약
4. 등장인물과 관계
5. 사건 또는 논리의 흐름
6. 화자의 의도와 감정
7. 문체와 톤
8. 번역 시 주의할 표현
9. 고유명사, 숫자, 날짜, 핵심 용어
10. 직역하면 어색해질 부분

아직 번역하지 말고 분석만 하라.

원문:
{source_text}""",
        ),
    ]


def draft_translation_messages(source_text: str, context: ContextPack) -> list[LLMMessage]:
    source_lang, target_lang = language_pair(context.direction)
    route = classify_route(context.direction, context.text_type)
    return [
        LLMMessage(
            role="system",
            content=f"너는 {source_lang}-{target_lang} 전문 번역가다. 정확성을 우선으로 초벌 번역을 만든다.",
        ),
        LLMMessage(
            role="user",
            content=f"""위 분석을 바탕으로 원문을 번역하라.

라우트: {route}

우선순위:
1. 원문 의미를 빠짐없이 반영한다.
2. 문단 구조를 유지한다.
3. 고유명사, 숫자, 시제, 인물관계를 정확히 반영한다.
4. 대화체라면 실제 대화처럼 자연스럽게 번역한다.
5. 서술체라면 글의 흐름과 문체를 살린다.
6. 과도한 의역은 피하되, 직역투는 만들지 않는다.

유형별 주의:
{route_guidance(context.direction, context.text_type)}

Context Pack:
{dump_context(context)}

원문:
{source_text}""",
        ),
    ]


def fluency_refinement_messages(translation: str, context: ContextPack) -> list[LLMMessage]:
    target_lang = language_pair(context.direction)[1]
    return [
        LLMMessage(
            role="system",
            content=f"너는 {target_lang} 유창성 편집자다. 의미를 보존하면서 목표 언어답게 다듬는다.",
        ),
        LLMMessage(
            role="user",
            content=f"""아래 번역문을 목표 언어 독자가 자연스럽게 읽을 수 있도록 다듬어라.

조건:
1. 원문 의미를 바꾸지 말 것
2. 누락이나 추가 정보를 만들지 말 것
3. 문체는 Context Pack과 일치시킬 것
4. 직역투를 제거할 것
5. 대화체라면 실제 사람이 말하는 듯하게, 서술체라면 글답게 다듬을 것
6. 불필요한 설명 없이 다듬은 번역문만 출력할 것

Context Pack:
{dump_context(context)}

초벌 번역:
{translation}""",
        ),
    ]


def style_control_messages(translation: str, context: ContextPack) -> list[LLMMessage]:
    route = classify_route(context.direction, context.text_type)
    return [
        LLMMessage(
            role="system",
            content="너는 번역 문체 편집자다. 의미는 유지하고 문체, 톤, 말투, 리듬만 보정한다.",
        ),
        LLMMessage(
            role="user",
            content=f"""아래 번역문을 문체 기준에 맞게 보정하라.

문체 기준:
- 유형: {context.text_type}
- 번역 방향: {context.direction}
- 라우트: {route}
- 톤: {context.style.voice or "Context Pack 기준"}
- 격식/문체: {context.style.register or "Context Pack 기준"}
- 인물관계: {characters_summary(context)}
- 독자에게 자연스럽게 읽히는 수준으로 조정할 것

주의:
1. 의미를 바꾸지 말 것
2. 과도하게 화려하게 만들지 말 것
3. 문체만 개선할 것
4. 불필요한 설명 없이 보정한 번역문만 출력할 것

번역문:
{translation}""",
        ),
    ]


def accuracy_verification_messages(
    source_text: str,
    translation: str,
    context: ContextPack,
) -> list[LLMMessage]:
    return [
        LLMMessage(
            role="system",
            content="너는 번역 검수자다. 원문과 번역문을 비교하여 정확성 오류와 문체 위험을 찾는다.",
        ),
        LLMMessage(
            role="user",
            content=f"""원문과 번역문을 비교해 오류를 찾아라.

검수 항목:
1. 누락된 의미
2. 추가된 의미
3. 오역
4. 어색한 직역투
5. 문체 불일치
6. 인물관계/말투 오류
7. 시제, 숫자, 고유명사 오류
8. 더 자연스러운 표현 제안

출력 형식:
- 문제 위치:
- 문제 유형:
- 설명:
- 수정 제안:
- 심각도: 높음/중간/낮음

Context Pack:
{dump_context(context)}

원문:
{source_text}

번역문:
{translation}""",
        ),
    ]


def final_revision_messages(translation: str, qa_report: str, context: ContextPack) -> list[LLMMessage]:
    return [
        LLMMessage(
            role="system",
            content="너는 대회 제출용 최종 번역 편집자다. 검수 결과를 반영하되 최종 번역문만 출력한다.",
        ),
        LLMMessage(
            role="user",
            content=f"""아래 번역문과 검수 결과를 반영하여 최종 제출용 번역문을 작성하라.

조건:
1. 원문 의미를 바꾸지 않는다.
2. 누락된 내용 없이 반영한다.
3. 목표 언어로 자연스럽게 읽히게 한다.
4. 문체를 일관되게 유지한다.
5. 불필요한 설명 없이 최종 번역문만 출력한다.

Context Pack:
{dump_context(context)}

번역문:
{translation}

검수 결과:
{qa_report}""",
        ),
    ]


def rubric_scoring_messages(source_text: str, translation: str, context: ContextPack) -> list[LLMMessage]:
    return [
        LLMMessage(
            role="system",
            content="너는 번역 대회 평가 기준 점검자다. 점수 예측보다 마지막 수정 포인트를 찾는 데 집중한다.",
        ),
        LLMMessage(
            role="user",
            content=f"""아래 번역문을 다음 기준으로 평가하라.

평가 기준:
1. 유창성 50점: 목표 언어로 자연스럽게 읽히는가
2. 문체성 30점: 문체, 구조, 어휘가 원문과 목적에 맞는가
3. 정확성 20점: 원문 내용이 빠짐없이 반영되었는가

각 항목별로:
- 점수
- 감점 이유
- 개선 제안
- 반드시 고쳐야 할 문장

Context Pack:
{dump_context(context)}

원문:
{source_text}

번역문:
{translation}""",
        ),
    ]


def dump_context(context: ContextPack) -> str:
    return json.dumps(context.to_dict(), ensure_ascii=False, indent=2)


def language_pair(direction: Direction) -> tuple[str, str]:
    if direction == "en-ko":
        return "영어", "한국어"
    return "한국어", "영어"


def route_guidance(direction: Direction, text_type: TextType) -> str:
    if direction == "en-ko" and text_type == "dialogue":
        return "\n".join(
            [
                "- 영어 대화를 한국어 대화처럼 자연스럽게 옮긴다.",
                "- 존댓말/반말, 말투의 거리감, 감정 표현을 인물관계에 맞춘다.",
                "- 영어식 표현과 대명사 반복을 줄이고 말의 호흡을 살린다.",
            ]
        )
    if direction == "en-ko" and text_type == "narrative":
        return "\n".join(
            [
                "- 영어 문장 구조를 그대로 끌고 오지 말고 한국어 글로 재구성한다.",
                "- 긴 문장은 필요하면 나누되 문단 흐름과 서술자의 시선을 유지한다.",
                "- 문학적/설명적 분위기를 보존한다.",
            ]
        )
    if direction == "ko-en" and text_type == "dialogue":
        return "\n".join(
            [
                "- 한국어에서 생략된 주어와 맥락을 영어에서 자연스럽게 복원한다.",
                "- 높임말은 영어의 정중한 표현으로 바꾸고 감정의 세기를 조절한다.",
                "- 한국어식 직역을 피하고 실제 영어 대화 표현을 사용한다.",
            ]
        )
    return "\n".join(
        [
            "- 한국어 문장을 영어 논리 구조에 맞게 재구성한다.",
            "- 주어와 동사, 시제, 관사, 단수/복수를 명확하게 처리한다.",
            "- 문장 간 논리 연결을 자연스럽게 만든다.",
        ]
    )


def characters_summary(context: ContextPack) -> str:
    if not context.characters:
        return "분석 결과에 따름"
    return "; ".join(
        f"{item.name or '인물'}({item.role or '역할 미상'}, {item.relationship or '관계 미상'}, {item.tone or '톤 미상'})"
        for item in context.characters
    )

