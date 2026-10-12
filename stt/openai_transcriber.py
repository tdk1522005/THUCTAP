import io
import os
import time
import wave

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class OpenAITranscriber:
    def __init__(self, client=None, model=None):
        self.model = model or os.getenv(
            "OPENAI_STT_MODEL",
            "gpt-4o-mini-transcribe",
        )
        self.client = client or OpenAI()

    @staticmethod
    def audio_to_wav_bytes(audio, sample_rate=16000):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        pcm_audio = (audio * 32767).astype("<i2")

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_audio.tobytes())

        buffer.seek(0)
        return buffer

    def transcribe(self, audio, sample_rate=16000):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio.size == 0:
            return {"text": "", "latency_ms": 0}

        wav_buffer = self.audio_to_wav_bytes(audio, sample_rate)
        wav_buffer.name = "audio.wav"

        start_time = time.perf_counter()
        response = self.client.audio.transcriptions.create(
            model=self.model,
            file=wav_buffer,
            language="en",
            response_format="text",
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        if isinstance(response, str):
            text = response.strip()
        else:
            text = getattr(response, "text", "").strip()

        return {
            "text": text,
            "latency_ms": latency_ms,
        }
