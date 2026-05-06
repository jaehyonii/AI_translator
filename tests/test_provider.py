import json
import unittest
from unittest.mock import patch

from ai_translator.models import LLMMessage
from ai_translator.provider import OpenAICompatibleProvider


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return b'{"choices":[{"message":{"content":"ok"}}]}'


class ProviderTests(unittest.TestCase):
    def _capture_payload(self, provider):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _FakeResponse()

        with patch("urllib.request.urlopen", fake_urlopen):
            result = provider.complete("test_step", [LLMMessage(role="user", content="hello")])

        self.assertEqual(result, "ok")
        return captured["payload"]

    def test_default_request_omits_temperature(self):
        provider = OpenAICompatibleProvider(api_key="test-key", model="test-model")

        payload = self._capture_payload(provider)

        self.assertNotIn("temperature", payload)

    def test_explicit_temperature_is_sent(self):
        provider = OpenAICompatibleProvider(api_key="test-key", model="test-model", temperature=0.2)

        payload = self._capture_payload(provider)

        self.assertEqual(payload["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
