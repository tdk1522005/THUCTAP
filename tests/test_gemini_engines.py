import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from stt.gemini_transcriber import GeminiTranscriber
from translation import gemini_translator
from translation.gemini_translator import GeminiTranslationEngine
from translation.gemini_tts import GeminiTTSEngine


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, list):
            result = self.response.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return self.response


class FakeClient:
    def __init__(self, response):
        self.models = FakeModels(response)


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"API error {code}")
        self.code = code


def test_gemini_transcriber_empty_audio_skips_api():
    client = FakeClient(SimpleNamespace(text="unexpected"))
    result = GeminiTranscriber(client=client).transcribe(np.array([], dtype=np.float32))
    assert result == {"text": "", "latency_ms": 0}
    assert client.models.calls == []


def test_gemini_transcriber_builds_wav_and_returns_text():
    client = FakeClient(SimpleNamespace(text="  Hello world.  "))
    result = GeminiTranscriber(client=client).transcribe(np.zeros(16000, dtype=np.float32))
    assert result["text"] == "Hello world."
    part = client.models.calls[0]["contents"][1]
    assert part.inline_data.mime_type == "audio/wav"


def test_gemini_translation_skips_empty_and_translates():
    client = FakeClient(SimpleNamespace(text=" Xin chào. "))
    engine = GeminiTranslationEngine(client=client)
    assert engine.translate(" ") == {"text": "", "latency_ms": 0}
    assert engine.translate("Hello.")["text"] == "Xin chào."
    assert len(client.models.calls) == 1


def test_gemini_translation_retries_503_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(gemini_translator.time, "sleep", sleeps.append)
    client = FakeClient([
        FakeAPIError(503),
        SimpleNamespace(text=" Xin chào. "),
    ])
    engine = GeminiTranslationEngine(client=client)
    result = engine.translate("Hello.")
    assert result["text"] == "Xin chào."
    assert len(client.models.calls) == 2
    assert sleeps == [0.5]


def test_gemini_translation_does_not_retry_non_503(monkeypatch):
    sleeps = []
    monkeypatch.setattr(gemini_translator.time, "sleep", sleeps.append)
    client = FakeClient([FakeAPIError(404)])
    engine = GeminiTranslationEngine(client=client)
    with pytest.raises(FakeAPIError):
        engine.translate("Hello.")
    assert len(client.models.calls) == 1
    assert sleeps == []


def test_gemini_tts_returns_valid_wav():
    pcm = b"\x00\x00" * 100
    response = SimpleNamespace(candidates=[
        SimpleNamespace(content=SimpleNamespace(parts=[
            SimpleNamespace(inline_data=SimpleNamespace(data=pcm))
        ]))
    ])
    engine = GeminiTTSEngine(client=FakeClient(response))
    result = engine.synthesize("Xin chào.")
    with wave.open(io.BytesIO(result["audio"]), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 100


def test_gemini_tts_empty_text_skips_api():
    client = FakeClient(SimpleNamespace(candidates=[]))
    assert GeminiTTSEngine(client=client).synthesize("") == {
        "audio": b"", "latency_ms": 0
    }
    assert client.models.calls == []
