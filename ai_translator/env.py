from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from a dotenv file into os.environ."""

    env_path = Path(path)
    if not env_path.exists():
        return {}
    if not env_path.is_file():
        raise ValueError(f"환경 파일 경로가 파일이 아닙니다: {env_path}")

    loaded: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = parse_dotenv_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if not key:
            raise ValueError(f"{env_path}:{line_number} 환경변수 이름이 비어 있습니다.")
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        raise ValueError(f"잘못된 .env 라인입니다: {line}")

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = strip_inline_comment(value.strip())
    return key, unquote(value)


def strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or value[index - 1].isspace():
                return value[:index].rstrip()
    return value


def unquote(value: str) -> str:
    if len(value) < 2:
        return value
    quote = value[0]
    if quote not in {"'", '"'} or value[-1] != quote:
        return value
    inner = value[1:-1]
    if quote == '"':
        return bytes(inner, "utf-8").decode("unicode_escape")
    return inner

