from unittest.mock import MagicMock

from translation.openai_tts import OpenAITTSEngine


def test_synthesize_returns_audio_and_latency():
    client = MagicMock()
    client.audio.speech.create.return_value.read.return_value = (
        b"fake-wav-audio"
    )

    engine = OpenAITTSEngine(client=client)
    result = engine.synthesize("Xin chào mọi người.")

    assert result["audio"] == b"fake-wav-audio"
    assert result["latency_ms"] >= 0

    client.audio.speech.create.assert_called_once_with(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input="Xin chào mọi người.",
        response_format="wav",
    )


def test_synthesize_empty_text_skips_api_call():
    client = MagicMock()
    engine = OpenAITTSEngine(client=client)

    result = engine.synthesize("   ")

    assert result == {
        "audio": b"",
        "latency_ms": 0,
    }
    client.audio.speech.create.assert_not_called()
