import io
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from stt.openai_transcriber import OpenAITranscriber


def test_empty_audio_does_not_call_api():
    client = MagicMock()
    transcriber = OpenAITranscriber(client=client)

    result = transcriber.transcribe(np.array([], dtype=np.float32))

    assert result == {"text": "", "latency_ms": 0}
    client.audio.transcriptions.create.assert_not_called()


def test_audio_to_wav_bytes_creates_valid_mono_wav():
    audio = np.zeros(1600, dtype=np.float32)

    buffer = OpenAITranscriber.audio_to_wav_bytes(audio)

    with wave.open(buffer, "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 1600


def test_transcribe_returns_expected_format():
    client = MagicMock()
    client.audio.transcriptions.create.return_value = (
        " Hello everyone. "
    )
    transcriber = OpenAITranscriber(
        client=client,
        model="gpt-4o-mini-transcribe",
    )

    result = transcriber.transcribe(
        np.zeros(1600, dtype=np.float32)
    )

    assert result["text"] == "Hello everyone."
    assert result["latency_ms"] >= 0
    client.audio.transcriptions.create.assert_called_once()

    call = client.audio.transcriptions.create.call_args
    assert call.kwargs["model"] == "gpt-4o-mini-transcribe"
    assert call.kwargs["language"] == "en"
    assert call.kwargs["response_format"] == "text"

    audio_file = call.kwargs["file"]
    assert audio_file.name == "audio.wav"
    assert audio_file.read(4) == b"RIFF"

def test_transcribe_propagates_api_error():
    client = MagicMock()
    client.audio.transcriptions.create.side_effect = RuntimeError(
        "API connection failed"
    )
    transcriber = OpenAITranscriber(client=client)

    audio = np.zeros(1600, dtype=np.float32)

    try:
        transcriber.transcribe(audio)
        assert False, "Expected RuntimeError to be raised"
    except RuntimeError as exc:
        assert str(exc) == "API connection failed"