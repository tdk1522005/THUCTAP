"""Thread/asyncio bridge for one Gemini Live session per Socket.IO client."""
import asyncio
import base64
import logging
import json
import os
import re
import threading
import time

from flask_socketio import emit

from stt.gemini_live_transcriber import GeminiLiveTranscriber

_live_sessions = {}
_FILLERS = re.compile(r"\b(um+|uh+|er+|ah+|hmm+)\b[,.]?\s*", re.I)
_REPEAT = re.compile(r"\b(\w+)(\s+\1\b)+", re.I)


def _clean_text(text):
    text = _FILLERS.sub("", text or "")
    text = _REPEAT.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bi\b", "I", text)
    return text[:1].upper() + text[1:] if text else text
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
        self.partial = ""
        self.history = []
        self.prediction_task = None
        self.prediction_enabled = os.getenv("GEMINI_PREDICTION_ENABLED", "true").lower() in {"1", "true", "yes"}
        self.prediction_model = os.getenv("GEMINI_PREDICTION_MODEL", "gemini-2.5-flash")

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
            if kind == "interim":
                self.partial = text
                if self.prediction_task and not self.prediction_task.done():
                    self.prediction_task.cancel()
                if self.prediction_enabled:
                    self.prediction_task = asyncio.create_task(self._delayed_predict(text))
                return
            if self.prediction_task and not self.prediction_task.done():
                self.prediction_task.cancel()
            cleaned = _clean_text(text)
            if cleaned:
                self.history.append(cleaned)
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
            except Exception as exc:
                logging.getLogger(__name__).exception("Live translation failed")
                socketio_emit(self.sid, "live_translation_error", {
                    "error": "Không dịch được câu vừa nhận: " + str(exc)
                })

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


    async def _delayed_predict(self, partial):
        try:
            await asyncio.sleep(float(os.getenv("GEMINI_PREDICTION_DEBOUNCE", "0.6")))
            snapshot = _clean_text(partial)
            if not snapshot or snapshot != _clean_text(self.partial):
                return
            prompt = (
                "You help with live speech. Given the conversation so far and an "
                "UNFINISHED sentence, return JSON with keys: "
                '"fixed": unfinished sentence with grammar/punctuation fixed (do not add words), '
                '"is_complete": true/false, "continuation": most likely next 3-8 words '
                '(empty if complete), "full": fixed + continuation as one natural sentence.\n\n'
                f"Previous sentences: {self.history[-3:]}\n"
                f"Unfinished sentence: {snapshot!r}"
            )
            from google.genai import types
            response = await self.engine.client.aio.models.generate_content(
                model=self.prediction_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.3
                ),
            )
            if snapshot != _clean_text(self.partial):
                return
            result = json.loads(response.text or "{}")
            socketio_emit(self.sid, "live_prediction", {
                "fixed": result.get("fixed", snapshot),
                "is_complete": bool(result.get("is_complete", False)),
                "continuation": result.get("continuation", ""),
                "full": result.get("full", snapshot),
            })
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.getLogger(__name__).exception("Live prediction failed")

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
            if self.prediction_task and not self.prediction_task.done():
                self.loop.call_soon_threadsafe(self.prediction_task.cancel)
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

    @socketio.on("live_tts_toggle")
    def handle_live_tts_toggle(data):
        sid = __import__("flask").request.sid
        with _live_lock:
            session = _live_sessions.get(sid)
        if session:
            session.tts_enabled = bool((data or {}).get("enabled", False))
            emit("live_tts_status", {"enabled": session.tts_enabled})

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
