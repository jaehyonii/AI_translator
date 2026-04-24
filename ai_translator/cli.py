from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .env import load_dotenv
from .pipeline import PipelineConfig, TranslationPipeline
from .provider import DryRunProvider, OpenAICompatibleProvider
from .report import render_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-translator",
        description="LLM 번역 아키텍처 기반 대회용 번역 보조 CLI",
    )
    parser.add_argument(
        "input",
        help="원문 파일 경로. '-'를 사용하면 표준 입력에서 읽습니다.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="결과 Markdown 저장 경로. 생략하면 표준 출력으로 출력합니다.",
    )
    parser.add_argument(
        "--direction",
        choices=["auto", "en-ko", "ko-en"],
        default="auto",
        help="번역 방향. 기본값은 자동 감지입니다.",
    )
    parser.add_argument(
        "--text-type",
        choices=["auto", "dialogue", "narrative"],
        default="auto",
        help="글 유형. 기본값은 자동 감지입니다.",
    )
    parser.add_argument(
        "--provider",
        choices=["dry-run", "openai-compatible"],
        default="dry-run",
        help="LLM Provider. 기본값은 API 호출 없는 dry-run입니다.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="openai-compatible Provider에서 사용할 모델명. 또는 LLM_MODEL 환경변수.",
    )
    parser.add_argument(
        "--api-key-env",
        default="LLM_API_KEY",
        help="API 키 환경변수 이름. 기본값은 LLM_API_KEY입니다.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI 호환 chat completions 엔드포인트.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="실행 전에 읽을 환경 파일 경로. 기본값은 .env입니다.",
    )
    parser.add_argument(
        "--no-env-file",
        action="store_true",
        help=".env 파일 자동 로드를 건너뜁니다.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="평가 기준 점검 단계를 건너뜁니다.",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="결과 Markdown에 각 단계 프롬프트와 출력을 포함합니다.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if not args.no_env_file:
            load_dotenv(args.env_file, override=False)
        source_text = read_input(args.input)
        provider = build_provider(args)
        pipeline = TranslationPipeline(
            provider,
            PipelineConfig(
                direction=args.direction,
                text_type=args.text_type,
                include_score=not args.no_score,
            ),
        )
        result = pipeline.run(source_text)
        report = render_markdown_report(result, include_prompts=args.show_prompts)
        write_output(report, args.output)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def read_input(path: str) -> str:
    if path == "-":
        text = sys.stdin.read()
    else:
        text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("입력 원문이 비어 있습니다.")
    return text


def build_provider(args: argparse.Namespace):
    if args.provider == "dry-run":
        return DryRunProvider()

    api_key = os.getenv(args.api_key_env, "").strip()
    model = (args.model or os.getenv("LLM_MODEL", "")).strip()
    base_url = (
        args.base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")
    ).strip()
    if not api_key:
        raise ValueError(f"{args.provider} 사용 시 {args.api_key_env} 환경변수가 필요합니다.")
    if not model:
        raise ValueError("openai-compatible Provider 사용 시 --model 또는 LLM_MODEL 환경변수가 필요합니다.")
    return OpenAICompatibleProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=args.temperature,
    )


def write_output(report: str, path: str | None) -> None:
    if not path:
        print(report, end="")
        return
    Path(path).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
