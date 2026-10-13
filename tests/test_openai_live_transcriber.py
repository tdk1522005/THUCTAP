import asyncio
import base64
from types import SimpleNamespace

from stt.openai_live_transcriber import OpenAILiveTranscriber


class FakeConnection:
    def __init__(self, events=None):
        self.events = list(events or [])
        self.sent = []

    async def send(self, event):
        self.sent.append(event)

    def __aiter__(self):
        self._events = iter(self.events)
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration


class FakeConnectionContext:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, *_args):
        return False


class FakeRealtime:
    def __init__(self, connection):
        self.connection = connection
        self.model = None

    def connect(self, model):
        self.model = model
        return FakeConnectionContext(self.connection)


class FakeClient:
    def __init__(self, connection):
        self.realtime = FakeRealtime(connection)


def test_openai_live_transcriber_accumulates_deltas_and_finalizes():
    async def run():
        connection = FakeConnection([
            SimpleNamespace(type="session.updated"),
            SimpleNamespace(type="conversation.item.input_audio_transcription.delta", delta="Hello"),
            SimpleNamespace(type="conversation.item.input_audio_transcription.delta", delta=" world"),
            SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript="Hello world."),
        ])
        engine = OpenAILiveTranscriber(client=FakeClient(connection))
        events = []

        async def on_transcript(kind, text):
            events.append((kind, text))

        await engine.start(on_transcript)
        await engine._receiver_task
        assert events == [
            ("interim", "Hello"),
            ("interim", "Hello world"),
            ("final", "Hello world."),
        ]
        assert connection.sent[0]["type"] == "session.update"
        assert connection.sent[0]["session"]["type"] == "transcription"
        await engine.close()

    asyncio.run(run())


def test_openai_live_transcriber_sends_pcm_as_base64():
    async def run():
        connection = FakeConnection([SimpleNamespace(type="session.updated")])
        engine = OpenAILiveTranscriber(client=FakeClient(connection))
        await engine.start(lambda *_: asyncio.sleep(0))
        pcm = b"\\x01\\x02" * 80
        await engine.send_audio(pcm)
        audio_event = connection.sent[-1]
        assert audio_event["type"] == "input_audio_buffer.append"
        assert base64.b64decode(audio_event["audio"]) == pcm
        await engine.close()

    asyncio.run(run())


def test_openai_live_transcriber_ignores_empty_audio_without_sending():
    async def run():
        connection = FakeConnection([SimpleNamespace(type="session.updated")])
        engine = OpenAILiveTranscriber(client=FakeClient(connection))
        await engine.start(lambda *_: asyncio.sleep(0))
        await engine.send_audio(b"")
        assert len(connection.sent) == 1
        await engine.close()

    asyncio.run(run())
