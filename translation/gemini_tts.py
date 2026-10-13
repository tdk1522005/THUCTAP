import io
import os
import time
import wave

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class GeminiTTSEngine:
    """Generate speech and return WAV bytes compatible with the existing browser player."""

    def __init__(self, client=None, model=None, voice=None):
        self.model = model or os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
        self.voice = voice or os.getenv("GEMINI_TTS_VOICE", "Kore")
        self.sample_rate = 24000
        self.client = client or genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    @staticmethod
    def pcm_to_wav(pcm_bytes, sample_rate=24000):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm_bytes)
        return buffer.getvalue()

    def synthesize(self, text):
        text = (text or "").strip()
        if not text:
            return {"audio": b"", "latency_ms": 0}

        started = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice
                        )
                    )
                ),
            ),
        )
        pcm = None
        for candidate in (getattr(response, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            for part in (getattr(content, "parts", None) or []):
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    pcm = inline.data
                    break
            if pcm:
                break
        if not pcm:
            raise RuntimeError("Gemini TTS không trả về dữ liệu âm thanh.")
        return {
            "audio": self.pcm_to_wav(pcm, self.sample_rate),
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
