import io
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class OpenAITTSEngine:
    def __init__(self, client=None, model=None, voice=None):
        self.model = model or os.getenv(
            "OPENAI_TTS_MODEL",
            "gpt-4o-mini-tts",
        )
        self.voice = voice or os.getenv(
            "OPENAI_TTS_VOICE",
            "alloy",
        )
        self.client = client or OpenAI()

    def synthesize(self, text):
        text = text.strip()

        if not text:
            return {
                "audio": b"",
                "latency_ms": 0,
            }

        start_time = time.perf_counter()

        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="wav",
        )

        audio_bytes = response.read()
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "audio": audio_bytes,
            "latency_ms": latency_ms,
        }
