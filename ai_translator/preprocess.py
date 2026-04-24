from __future__ import annotations

import re

from .models import Direction, ProcessedSource, TextType


HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
LATIN_RE = re.compile(r"[A-Za-z]")
PROPER_NOUN_RE = re.compile(r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)*\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:%|[a-zA-Z]+)?\b")
DATE_RE = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}(?:[-/.]\d{2,4})?)\b"
)
SPEAKER_RE = re.compile(r"^\s*(?:[-–—]|[A-Za-z가-힣]{1,20}\s*:)")
TITLECASE_STOPWORDS = {
    "A",
    "An",
    "And",
    "Are",
    "As",
    "At",
    "But",
    "By",
    "For",
    "From",
    "He",
    "Her",
    "His",
    "I",
    "If",
    "In",
    "It",
    "Its",
    "On",
    "Or",
    "She",
    "That",
    "The",
    "They",
    "This",
    "To",
    "We",
    "When",
    "Where",
    "You",
}


def preprocess_source(text: str) -> ProcessedSource:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    dialogue_lines = [line for line in lines if is_dialogue_line(line)]
    notable_terms = extract_notable_terms(normalized)
    return ProcessedSource(
        original_text=normalized,
        paragraphs=paragraphs,
        dialogue_lines=dialogue_lines,
        notable_terms=notable_terms,
    )


def is_dialogue_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if SPEAKER_RE.search(stripped):
        return True
    quote_marks = stripped.count('"') + stripped.count("'") + stripped.count("“") + stripped.count("”")
    korean_quote_marks = stripped.count("“") + stripped.count("”") + stripped.count("‘") + stripped.count("’")
    return quote_marks >= 2 or korean_quote_marks >= 2


def extract_notable_terms(text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in (DATE_RE, NUMBER_RE, ACRONYM_RE):
        candidates.extend(match.group(0) for match in pattern.finditer(text))
    for match in PROPER_NOUN_RE.finditer(text):
        value = match.group(0)
        if value not in TITLECASE_STOPWORDS:
            candidates.append(value)
    return dedupe_preserve_order(item.strip() for item in candidates if item.strip())


def detect_direction(text: str) -> Direction:
    hangul_count = len(HANGUL_RE.findall(text))
    latin_count = len(LATIN_RE.findall(text))
    if hangul_count > latin_count:
        return "ko-en"
    return "en-ko"


def detect_text_type(processed: ProcessedSource) -> TextType:
    total_lines = max(1, len([line for line in processed.original_text.split("\n") if line.strip()]))
    dialogue_ratio = len(processed.dialogue_lines) / total_lines
    quote_count = sum(processed.original_text.count(mark) for mark in ['"', "“", "”", "‘", "’"])
    if dialogue_ratio >= 0.25 or quote_count >= 2:
        return "dialogue"
    return "narrative"


def render_preprocess_report(processed: ProcessedSource) -> str:
    paragraph_lines = [
        f"[문단 {index}]\n{paragraph}" for index, paragraph in enumerate(processed.paragraphs, start=1)
    ]
    dialogue = "\n".join(f"- {line}" for line in processed.dialogue_lines[:20]) or "- 감지된 대화 행 없음"
    notables = "\n".join(f"- {term}" for term in processed.notable_terms[:40]) or "- 자동 감지된 항목 없음"
    return "\n\n".join(
        [
            "\n\n".join(paragraph_lines) if paragraph_lines else "[문단 1]\n",
            "[감지된 대화 행]\n" + dialogue,
            "[주의 요소 후보]\n" + notables,
        ]
    )


def dedupe_preserve_order(items: object) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = str(item)
        if key not in seen:
            seen.add(key)
            output.append(key)
    return output
