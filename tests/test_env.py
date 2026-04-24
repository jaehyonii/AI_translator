import os
import unittest
from pathlib import Path

from ai_translator.env import load_dotenv, parse_dotenv_line


class EnvTests(unittest.TestCase):
    def test_parse_dotenv_line(self):
        self.assertEqual(parse_dotenv_line("LLM_MODEL=test-model"), ("LLM_MODEL", "test-model"))
        self.assertEqual(parse_dotenv_line("export LLM_API_KEY='abc#123'"), ("LLM_API_KEY", "abc#123"))
        self.assertEqual(parse_dotenv_line('LLM_BASE_URL="https://example.com" # comment'), ("LLM_BASE_URL", "https://example.com"))
        self.assertIsNone(parse_dotenv_line("# comment"))

    def test_load_dotenv_does_not_override_existing_env_by_default(self):
        env_path = Path(__file__).with_name("fixtures") / "sample.env"
        previous = os.environ.get("AI_TRANSLATOR_TEST_VALUE")
        os.environ["AI_TRANSLATOR_TEST_VALUE"] = "from-env"
        try:
            loaded = load_dotenv(env_path)
            self.assertEqual(loaded["AI_TRANSLATOR_TEST_VALUE"], "from-file")
            self.assertEqual(os.environ["AI_TRANSLATOR_TEST_VALUE"], "from-env")
        finally:
            if previous is None:
                os.environ.pop("AI_TRANSLATOR_TEST_VALUE", None)
            else:
                os.environ["AI_TRANSLATOR_TEST_VALUE"] = previous


if __name__ == "__main__":
    unittest.main()
