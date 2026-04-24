# AI Translator

`llm_translation_architecture.md`의 설계를 코드로 옮긴 대회용 LLM 번역 보조 CLI입니다.

핵심 흐름은 다음과 같습니다.

```text
전처리
→ Context Analyzer
→ Draft Translator
→ Fluency Refiner
→ Style Controller
→ Accuracy Verifier
→ Final Editor
→ Rubric Scorer
```

## 빠른 실행

API 호출 없이 프롬프트와 파이프라인 흐름만 확인:

```powershell
python -m ai_translator .\input.txt --provider dry-run --show-prompts
```

결과를 Markdown 파일로 저장:

```powershell
python -m ai_translator .\input.txt -o .\result.md --provider dry-run --show-prompts
```

실제 LLM 호출은 OpenAI 호환 `chat/completions` 엔드포인트를 사용합니다.

`.env` 파일에 환경변수를 저장할 수 있습니다. 실제 `.env`는 `.gitignore`에 포함되어 추적되지 않습니다.

```dotenv
LLM_API_KEY=your-api-key
LLM_MODEL=your-model
LLM_BASE_URL=https://api.openai.com/v1/chat/completions
```

```powershell
python -m ai_translator .\input.txt --provider openai-compatible -o .\result.md
```

기본 환경파일 경로는 `.env`입니다. 다른 파일을 쓰려면 `--env-file`을 지정합니다.

```powershell
python -m ai_translator .\input.txt --provider openai-compatible --env-file .\.env.local
```

## 주요 옵션

- `--direction auto|en-ko|ko-en`: 번역 방향. 기본값은 `auto`.
- `--text-type auto|dialogue|narrative`: 대화체/서술체. 기본값은 `auto`.
- `--provider dry-run|openai-compatible`: LLM 호출 방식. 기본값은 `dry-run`.
- `--env-file PATH`: 실행 전에 읽을 환경 파일. 기본값은 `.env`.
- `--no-env-file`: `.env` 자동 로드를 건너뜀.
- `--show-prompts`: 각 Agent 프롬프트와 출력을 결과 Markdown에 포함.
- `--no-score`: Rubric Scorer 단계를 생략.

## 테스트

```powershell
python -m unittest discover -s tests
```
