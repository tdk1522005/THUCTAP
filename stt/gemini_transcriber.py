import io
import os
import time
import wave

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class GeminiTranscriber:
    """Transcribe a completed audio chunk with Gemini; no API call for empty audio."""

    def __init__(self, client=None, model=None):
        self.model = model or os.getenv(
            "GEMINI_STT_MODEL", "gemini-3.5-transcribe-live"
        )
        self.client = client or genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    @staticmethod
    def audio_to_wav_bytes(audio, sample_rate=16000):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
        return buffer.getvalue()

    def transcribe(self, audio, sample_rate=16000):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return {"text": "", "latency_ms": 0}

        wav_bytes = self.audio_to_wav_bytes(audio, sample_rate)
        started = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                "Transcribe the spoken English audio verbatim. Return only the "
                "recognized words; do not translate, summarize, or add commentary.",
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            ],
        )
        return {
            "text": (getattr(response, "text", None) or "").strip(),
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
