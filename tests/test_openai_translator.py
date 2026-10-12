from types import SimpleNamespace
from unittest.mock import MagicMock

from translation.openai_translator import OpenAITranslationEngine


def test_empty_input_does_not_call_api():
    client = MagicMock()
    engine = OpenAITranslationEngine(client=client)

    result = engine.translate("   ")

    assert result == {"text": "", "latency_ms": 0}
    client.responses.create.assert_not_called()


def test_translate_calls_api_and_returns_expected_format():
    client = MagicMock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="Tôi muốn đi đến sân bay."
    )
    engine = OpenAITranslationEngine(
        client=client,
        model="gpt-4.1-mini",
    )

    result = engine.translate("  I want to go to the airport.  ")

    assert result["text"] == "Tôi muốn đi đến sân bay."
    assert result["latency_ms"] >= 0
    client.responses.create.assert_called_once_with(
        model="gpt-4.1-mini",
        instructions=(
            "Translate the user's English text into natural, "
            "accurate Vietnamese. Preserve the original meaning. "
            "Return only the Vietnamese translation."
        ),
        input="I want to go to the airport.",
        max_output_tokens=500,
    )
