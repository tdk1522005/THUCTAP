import asyncio
from types import SimpleNamespace

import pytest

from stt.gemini_live_transcriber import GeminiLiveTranscriber


class FakeLiveSession:
    def __init__(self, messages=None):
        self.messages = messages or []
        self.sent = []

    async def send_realtime_input(self, **kwargs):
        self.sent.append(kwargs)

    async def receive(self):
        for message in self.messages:
            yield message


class FakeLiveConnection:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *args):
        return False


class FakeLive:
    def __init__(self, session):
        self.session = session
        self.last_config = None

    def connect(self, **kwargs):
        self.last_config = kwargs
        return FakeLiveConnection(self.session)


class FakeClient:
    def __init__(self, session):
        self.aio = SimpleNamespace(live=FakeLive(session))


def test_live_transcriber_sends_16khz_pcm():
    async def scenario():
        session = FakeLiveSession()
        engine = GeminiLiveTranscriber(client=FakeClient(session))
        await engine.start(lambda kind, text: asyncio.sleep(0))
        await engine.send_audio(b"\\x00\\x00" * 160)
        assert session.sent[0]["audio"].mime_type == "audio/pcm;rate=16000"
        assert session.sent[0]["audio"].data == b"\\x00\\x00" * 160
        await engine.close()
    asyncio.run(scenario())


def test_live_transcriber_emits_interim_and_final_events():
    async def scenario():
        messages = [
            SimpleNamespace(server_content=SimpleNamespace(
                interim_input_transcription=SimpleNamespace(text="hello"),
                input_transcription=None,
            )),
            SimpleNamespace(server_content=SimpleNamespace(
                interim_input_transcription=None,
                input_transcription=SimpleNamespace(text="Hello there."),
            )),
        ]
        engine = GeminiLiveTranscriber(client=FakeClient(FakeLiveSession(messages)))
        events = []

        async def on_transcript(kind, text):
            events.append((kind, text))

        await engine.start(on_transcript)
        if engine._receiver_task:
            await engine._receiver_task
        assert events == [("interim", "hello"), ("final", "Hello there.")]
        await engine.close()
    asyncio.run(scenario())


def test_live_transcriber_rejects_audio_when_not_started():
    async def scenario():
        engine = GeminiLiveTranscriber(client=FakeClient(FakeLiveSession()))
        with pytest.raises(RuntimeError):
            await engine.send_audio(b"\\x00\\x00")
    asyncio.run(scenario())
