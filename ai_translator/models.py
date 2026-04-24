from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Direction = Literal["en-ko", "ko-en"]
DirectionOption = Literal["auto", "en-ko", "ko-en"]
TextType = Literal["dialogue", "narrative"]
TextTypeOption = Literal["auto", "dialogue", "narrative"]


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class Character:
    name: str = ""
    role: str = ""
    relationship: str = ""
    tone: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Character":
        return cls(
            name=str(value.get("name", "") or ""),
            role=str(value.get("role", "") or ""),
            relationship=str(value.get("relationship", "") or ""),
            tone=str(value.get("tone", "") or ""),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "role": self.role,
            "relationship": self.relationship,
            "tone": self.tone,
        }


@dataclass
class Style:
    register: str = ""
    voice: str = ""
    sentence_length: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Style":
        return cls(
            register=str(value.get("register", "") or ""),
            voice=str(value.get("voice", "") or ""),
            sentence_length=str(value.get("sentence_length", "") or ""),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "register": self.register,
            "voice": self.voice,
            "sentence_length": self.sentence_length,
        }


@dataclass
class KeyTerm:
    source: str = ""
    translation: str = ""
    note: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "KeyTerm":
        return cls(
            source=str(value.get("source", "") or ""),
            translation=str(value.get("translation", "") or ""),
            note=str(value.get("note", "") or ""),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "translation": self.translation,
            "note": self.note,
        }


@dataclass
class ContextPack:
    direction: Direction
    text_type: TextType
    summary: str = ""
    characters: list[Character] = field(default_factory=list)
    plot_flow: list[str] = field(default_factory=list)
    style: Style = field(default_factory=Style)
    key_terms: list[KeyTerm] = field(default_factory=list)
    risk_points: list[str] = field(default_factory=list)
    raw_analysis: str = ""

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        *,
        default_direction: Direction,
        default_text_type: TextType,
        raw_analysis: str = "",
    ) -> "ContextPack":
        direction = normalize_direction(value.get("direction"), default_direction)
        text_type = normalize_text_type(value.get("text_type"), default_text_type)

        characters_raw = value.get("characters", [])
        if not isinstance(characters_raw, list):
            characters_raw = []

        key_terms_raw = value.get("key_terms", [])
        if not isinstance(key_terms_raw, list):
            key_terms_raw = []

        plot_flow = value.get("plot_flow", [])
        if not isinstance(plot_flow, list):
            plot_flow = [str(plot_flow)]

        risk_points = value.get("risk_points", [])
        if not isinstance(risk_points, list):
            risk_points = [str(risk_points)]

        style_raw = value.get("style", {})
        if not isinstance(style_raw, dict):
            style_raw = {}

        return cls(
            direction=direction,
            text_type=text_type,
            summary=str(value.get("summary", "") or ""),
            characters=[
                Character.from_mapping(item)
                for item in characters_raw
                if isinstance(item, dict)
            ],
            plot_flow=[str(item) for item in plot_flow if str(item).strip()],
            style=Style.from_mapping(style_raw),
            key_terms=[
                KeyTerm.from_mapping(item)
                for item in key_terms_raw
                if isinstance(item, dict)
            ],
            risk_points=[str(item) for item in risk_points if str(item).strip()],
            raw_analysis=raw_analysis,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "text_type": self.text_type,
            "summary": self.summary,
            "characters": [item.to_dict() for item in self.characters],
            "plot_flow": self.plot_flow,
            "style": self.style.to_dict(),
            "key_terms": [item.to_dict() for item in self.key_terms],
            "risk_points": self.risk_points,
        }


@dataclass
class ProcessedSource:
    original_text: str
    paragraphs: list[str]
    dialogue_lines: list[str]
    notable_terms: list[str]


@dataclass
class StepResult:
    name: str
    messages: list[LLMMessage]
    output: str


def normalize_direction(value: Any, default: Direction = "en-ko") -> Direction:
    text = str(value or "").strip().lower()
    if text in {"en-ko", "en_to_ko", "english to korean", "english->korean"}:
        return "en-ko"
    if text in {"영한", "영어-한국어", "영어→한국어", "영어에서 한국어"}:
        return "en-ko"
    if "english" in text and "korean" in text and text.index("english") < text.index("korean"):
        return "en-ko"
    if text in {"ko-en", "ko_to_en", "korean to english", "korean->english"}:
        return "ko-en"
    if text in {"한영", "한국어-영어", "한국어→영어", "한국어에서 영어"}:
        return "ko-en"
    if "korean" in text and "english" in text and text.index("korean") < text.index("english"):
        return "ko-en"
    return default


def normalize_text_type(value: Any, default: TextType = "narrative") -> TextType:
    text = str(value or "").strip().lower()
    if text in {"dialogue", "dialog", "conversation", "대화", "대화체"}:
        return "dialogue"
    if "dialogue" in text or "conversation" in text or "대화" in text:
        return "dialogue"
    if text in {"narrative", "prose", "description", "서술", "서술체"}:
        return "narrative"
    if "narrative" in text or "prose" in text or "서술" in text:
        return "narrative"
    return default

