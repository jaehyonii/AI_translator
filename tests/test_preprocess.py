import unittest

from ai_translator.preprocess import detect_direction, detect_text_type, preprocess_source


class PreprocessTests(unittest.TestCase):
    def test_detect_direction_defaults_to_english_to_korean_for_english_text(self):
        self.assertEqual(detect_direction("I didn't say you were wrong."), "en-ko")

    def test_detect_direction_detects_korean_to_english(self):
        self.assertEqual(detect_direction("잠깐 얘기 좀 할 수 있을까요?"), "ko-en")

    def test_detect_text_type_dialogue_from_speaker_lines(self):
        processed = preprocess_source("John: Are you serious?\nMary: I am.")
        self.assertEqual(detect_text_type(processed), "dialogue")

    def test_extracts_notable_terms(self):
        processed = preprocess_source("John met Mary on 2026-04-24 with 3 reports.")
        self.assertIn("John", processed.notable_terms)
        self.assertIn("Mary", processed.notable_terms)
        self.assertIn("2026-04-24", processed.notable_terms)
        self.assertIn("3", processed.notable_terms)


if __name__ == "__main__":
    unittest.main()

