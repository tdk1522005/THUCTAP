"""Gemini Live API session wrapper for low-latency interim/final transcription.

The wrapper is transport-agnostic so the web server can own one session per client.
Audio input must be mono signed 16-bit PCM at 16 kHz.
"""
import asyncio
import os
from collections.abc import Awaitable, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

TranscriptCallback = Callable[[str, str], Awaitable[None]]


class GeminiLiveTranscriber:
    def __init__(self, client=None, model=None, language="en-US"):
        self.model = model or os.getenv(
            "GEMINI_LIVE_MODEL", "gemini-3.5-transcribe-live"
        )
        self.language = language
        self.client = client or genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self._session = None
        self._receiver_task = None
        self._callback = None
        self._closed = False

    async def start(self, on_transcript: TranscriptCallback):
        """Connect and start receiving events. Callback receives (kind, text)."""
        if self._session is not None:
            raise RuntimeError("Gemini Live session is already started.")
        self._callback = on_transcript
        self._closed = False
        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=[self.language],
                mode="VERBATIM",
            ),
        )
        self._connection = self.client.aio.live.connect(
            model=self.model, config=config
        )
        self._session = await self._connection.__aenter__()
        self._receiver_task = asyncio.create_task(self._receive_loop())
        return self

    async def send_audio(self, pcm16_bytes: bytes):
        if self._session is None or self._closed:
            raise RuntimeError("Gemini Live session is not active.")
        if not isinstance(pcm16_bytes, bytes) or not pcm16_bytes:
            return
        await self._session.send_realtime_input(
            audio=types.Blob(
                data=pcm16_bytes,
                mime_type="audio/pcm;rate=16000",
            )
        )

    async def _receive_loop(self):
        try:
            async for message in self._session.receive():
                server_content = getattr(message, "server_content", None)
                if server_content is None:
                    continue
                interim = getattr(server_content, "interim_input_transcription", None)
                if interim and getattr(interim, "text", None):
                    await self._callback("interim", interim.text)
                final = getattr(server_content, "input_transcription", None)
                if final and getattr(final, "text", None):
                    await self._callback("final", final.text)
        except asyncio.CancelledError:
            raise

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            self._receiver_task = None
        if getattr(self, "_connection", None) is not None:
            await self._connection.__aexit__(None, None, None)
        self._session = None
        self._connection = None
