"""Thread/asyncio bridge for one Gemini Live session per Socket.IO client."""
import asyncio
import base64
import threading
import time

from flask import current_app
from flask_socketio import emit

from stt.gemini_live_transcriber import GeminiLiveTranscriber

_live_sessions = {}
_live_lock = threading.Lock()


class _LiveSession:
    def __init__(self, sid, translator, tts_engine):
        self.sid = sid
        self.translator = translator
        self.tts_engine = tts_engine
        self.loop = asyncio.new_event_loop()
        self.queue = None
        self.engine = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.ready = threading.Event()
        self.error = None
        self.tts_enabled = False
        self.closed = False

    def start(self, tts_enabled=False):
        self.tts_enabled = bool(tts_enabled)
        self.thread.start()
        if not self.ready.wait(timeout=25):
            self.stop()
            raise TimeoutError("Hết thời gian mở phiên Gemini Live.")
        if self.error:
            raise RuntimeError(str(self.error))

    def _run(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        except Exception as exc:
            self.error = exc
            self.ready.set()
            socketio_emit(self.sid, "live_error", {"error": str(exc)})
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.close()

    async def _serve(self):
        self.queue = asyncio.Queue(maxsize=40)
        self.engine = GeminiLiveTranscriber()

        async def on_transcript(kind, text):
            text = (text or "").strip()
            if not text:
                return
            socketio_emit(self.sid, "live_transcript", {"kind": kind, "text": text})
            if kind != "final":
                return
            started = time.perf_counter()
            try:
                translated = await asyncio.to_thread(self.translator.translate, text)
                vietnamese = (translated.get("text") or "").strip()
                audio_b64 = None
                tts_ms = 0
                if self.tts_enabled and vietnamese:
                    spoken = await asyncio.to_thread(self.tts_engine.synthesize, vietnamese)
                    tts_ms = spoken.get("latency_ms", 0)
                    if spoken.get("audio"):
                        audio_b64 = base64.b64encode(spoken["audio"]).decode("ascii")
                socketio_emit(self.sid, "live_translation", {
                    "english": text,
                    "vietnamese": vietnamese,
                    "translation_ms": translated.get("latency_ms", 0),
                    "tts_ms": tts_ms,
                    "total_ms": (time.perf_counter() - started) * 1000,
                    "audio_base64": audio_b64,
                })
            except Exception:
                current_app.logger.exception("Live translation failed")

        try:
            await self.engine.start(on_transcript)
            self.ready.set()
            socketio_emit(self.sid, "live_ready", {"ready": True})
            while True:
                item = await self.queue.get()
                if item is None:
                    break
                await self.engine.send_audio(item)
        finally:
            if self.engine:
                await self.engine.close()

    def send_audio(self, data):
        if self.closed or not self.loop.is_running() or self.queue is None:
            return False
        future = asyncio.run_coroutine_threadsafe(self.queue.put(data), self.loop)
        try:
            future.result(timeout=2)
            return True
        except Exception:
            future.cancel()
            return False

    def stop(self):
        if self.closed:
            return
        self.closed = True
        if self.loop.is_running() and self.queue is not None:
            asyncio.run_coroutine_threadsafe(self.queue.put(None), self.loop)
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=8)


def socketio_emit(sid, event, payload):
    # Socket.IO emit is safe from the session worker thread when the target SID is explicit.
    socketio.emit(event, payload, to=sid)


def configure_live_socketio(socketio_instance, translator, tts_engine):
    global socketio
    socketio = socketio_instance

    @socketio.on("live_start")
    def handle_live_start(data=None):
        sid = __import__("flask").request.sid
        with _live_lock:
            if sid in _live_sessions:
                emit("live_error", {"error": "Phiên realtime đã chạy."})
                return
            session = _LiveSession(sid, translator, tts_engine)
            _live_sessions[sid] = session
        try:
            session.start((data or {}).get("tts_enabled", False))
        except Exception as exc:
            with _live_lock:
                _live_sessions.pop(sid, None)
            session.stop()
            emit("live_error", {"error": "Không mở được Gemini Live: " + str(exc)})

    @socketio.on("live_audio")
    def handle_live_audio(data):
        sid = __import__("flask").request.sid
        if not isinstance(data, dict) or not isinstance(data.get("audio"), str):
            emit("live_error", {"error": "Gói audio realtime không hợp lệ."})
            return
        try:
            pcm = base64.b64decode(data["audio"], validate=True)
        except (ValueError, TypeError):
            emit("live_error", {"error": "Không giải mã được audio realtime."})
            return
        with _live_lock:
            session = _live_sessions.get(sid)
        if session and pcm:
            if not session.send_audio(pcm):
                emit("live_error", {"error": "Hàng đợi audio đầy hoặc phiên đã dừng."})

    @socketio.on("live_stop")
    def handle_live_stop():
        sid = __import__("flask").request.sid
        with _live_lock:
            session = _live_sessions.pop(sid, None)
        if session:
            session.stop()
        emit("live_stopped", {"stopped": True})

    @socketio.on("disconnect")
    def handle_live_disconnect():
        sid = __import__("flask").request.sid
        with _live_lock:
            session = _live_sessions.pop(sid, None)
        if session:
            session.stop()


socketio = None
