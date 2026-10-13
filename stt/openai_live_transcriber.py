"""OpenAI Realtime transcription session for streaming English audio."""
import asyncio
import base64
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


class OpenAILiveTranscriber:
    """Streams 16 kHz mono PCM to OpenAI and emits interim/final transcript text."""

    def __init__(self, client=None, model=None, language="en"):
        self.model = model or os.getenv("OPENAI_LIVE_MODEL", "gpt-live-transcribe")
        self.language = language
        self.client = client or AsyncOpenAI()
        self.connection = None
        self._context = None
        self._receiver_task = None
        self._callback = None
        self._closed = False
        self._ready = asyncio.Event()
        self._error = None
        self._partial_text = ""

    async def start(self, on_transcript):
        if self.connection is not None:
            raise RuntimeError("OpenAI Realtime session is already started.")
        self._callback = on_transcript
        self._closed = False
        self._ready.clear()
        self._error = None
        self._partial_text = ""
        self._context = self.client.realtime.connect(model=self.model)
        self.connection = await self._context.__aenter__()
        await self.connection.send({
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {
                            "model": self.model,
                            "languages": [self.language],
                        },
                        "turn_detection": None,
                    }
                },
            },
        })
        self._receiver_task = asyncio.create_task(self._receive_loop())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=15)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise TimeoutError("OpenAI Realtime did not confirm the transcription session.") from exc
        if self._error:
            error = self._error
            await self.close()
            raise RuntimeError(error)
        return self

    async def send_audio(self, pcm16_bytes):
        if self.connection is None or self._closed:
            raise RuntimeError("OpenAI Realtime session is not active.")
        if not isinstance(pcm16_bytes, bytes) or not pcm16_bytes:
            return
        await self.connection.send({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16_bytes).decode("ascii"),
        })

    async def _receive_loop(self):
        try:
            async for event in self.connection:
                event_type = getattr(event, "type", "")
                if event_type in {"session.updated", "transcription_session.updated"}:
                    self._ready.set()
                elif event_type == "conversation.item.input_audio_transcription.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        self._partial_text += delta
                        await self._callback("interim", self._partial_text)
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = getattr(event, "transcript", "")
                    if transcript:
                        self._partial_text = ""
                        await self._callback("final", transcript)
                elif event_type in {"error", "conversation.item.input_audio_transcription.failed"}:
                    error = getattr(event, "error", None)
                    message = getattr(error, "message", None) or getattr(event, "message", None) or str(event)
                    self._error = message
                    self._ready.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                self._error = str(exc)
                self._ready.set()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            self._receiver_task = None
        if self._context is not None:
            try:
                await self._context.__aexit__(None, None, None)
            finally:
                self._context = None
                self.connection = None
