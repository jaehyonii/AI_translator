import unittest

from ai_translator.pipeline import PipelineConfig, TranslationPipeline, extract_json_object, parse_context_pack
from ai_translator.prompts import classify_route
from ai_translator.provider import DryRunProvider


class PipelineTests(unittest.TestCase):
    def test_classify_route(self):
        self.assertEqual(classify_route("en-ko", "dialogue"), "영한번역 - 대화체")
        self.assertEqual(classify_route("ko-en", "narrative"), "한영번역 - 서술체")

    def test_parse_context_pack_from_fenced_json(self):
        raw = """```json
{"direction":"ko-en","text_type":"dialogue","summary":"test"}
```"""
        context = parse_context_pack(raw, default_direction="en-ko", default_text_type="narrative")
        self.assertEqual(context.direction, "ko-en")
        self.assertEqual(context.text_type, "dialogue")
        self.assertEqual(context.summary, "test")

    def test_extract_json_object_from_surrounding_text(self):
        self.assertEqual(extract_json_object('prefix {"summary":"ok"} suffix'), {"summary": "ok"})

    def test_dry_run_pipeline_completes(self):
        pipeline = TranslationPipeline(
            DryRunProvider(),
            PipelineConfig(direction="en-ko", text_type="dialogue", include_score=False),
        )
        result = pipeline.run('John: "Are you serious?"')
        self.assertEqual(result.route, "영한번역 - 대화체")
        self.assertIn("DRY RUN", result.final_translation)
        self.assertEqual([step.name for step in result.steps][-1], "final_revision")


if __name__ == "__main__":
    unittest.main()

