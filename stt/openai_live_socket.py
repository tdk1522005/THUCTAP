"""Thread/asyncio bridge for OpenAI Realtime transcription and ordered translation."""
import asyncio
import base64
import logging
import re
import threading
import time

from flask_socketio import emit

from stt.openai_live_transcriber import OpenAILiveTranscriber

_live_sessions = {}
_live_lock = threading.Lock()
_FILLERS = re.compile(r"\b(um+|uh+|er+|ah+|hmm+)\b[,.]?\s*", re.I)
_REPEAT = re.compile(r"\b(\w+)(\s+\1\b)+", re.I)


def _clean_text(text):
    text = _FILLERS.sub("", text or "")
    text = _REPEAT.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bi\b", "I", text)
    return text[:1].upper() + text[1:] if text else text


class _LiveSession:
    def __init__(self, sid, translator, tts_engine):
        self.sid = sid
        self.translator = translator
        self.tts_engine = tts_engine
        self.loop = asyncio.new_event_loop()
        self.audio_queue = None
        self.translation_queue = None
        self.translation_worker = None
        self.engine = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.ready = threading.Event()
        self.error = None
        self.tts_enabled = False
        self.closed = False
        self.history = []
        self.translation_tasks = set()

    def start(self, tts_enabled=False):
        self.tts_enabled = bool(tts_enabled)
        self.thread.start()
        if not self.ready.wait(timeout=22):
            self.stop()
            raise TimeoutError("Hết thời gian mở phiên OpenAI Realtime.")
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
        self.audio_queue = asyncio.Queue(maxsize=80)
        self.translation_queue = asyncio.Queue()
        self.engine = OpenAILiveTranscriber()

        async def on_transcript(kind, text):
            text = (text or "").strip()
            if not text:
                return
            socketio_emit(self.sid, "live_transcript", {"kind": kind, "text": text})
            if kind == "final":
                cleaned = _clean_text(text)
                if cleaned:
                    self.history.append(cleaned)
                await self.translation_queue.put(cleaned or text)

        try:
            self.translation_worker = asyncio.create_task(self._translation_worker())
            await self.engine.start(on_transcript)
            self.ready.set()
            socketio_emit(self.sid, "live_ready", {"ready": True})
            while True:
                item = await self.audio_queue.get()
                if item is None:
                    break
                if item == "__commit__":
                    await self.engine.commit_audio()
                else:
                    await self.engine.send_audio(item)
        finally:
            if self.translation_queue is not None and self.translation_worker and not self.translation_worker.done():
                try:
                    await asyncio.wait_for(self.translation_queue.join(), timeout=8)
                except asyncio.TimeoutError:
                    logging.getLogger(__name__).warning("Timed out draining OpenAI translation queue on stop")
            if self.translation_worker and not self.translation_worker.done():
                self.translation_worker.cancel()
                await asyncio.gather(self.translation_worker, return_exceptions=True)
            if self.engine:
                await self.engine.close()

    async def _translation_worker(self):
        while True:
            text = await self.translation_queue.get()
            try:
                if text:
                    await self._translate_final(text)
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.getLogger(__name__).exception("Queued OpenAI translation failed")
            finally:
                self.translation_queue.task_done()

    async def _translate_final(self, text):
        started = time.perf_counter()
        try:
            translated = await asyncio.to_thread(self.translator.translate, text)
            vietnamese = (translated.get("text") or "").strip()
            if not vietnamese:
                raise RuntimeError("OpenAI trả về bản dịch rỗng; câu tiếng Anh vẫn được giữ lại.")
            socketio_emit(self.sid, "live_translation", {
                "english": text,
                "vietnamese": vietnamese,
                "translation_ms": translated.get("latency_ms", 0),
                "total_ms": (time.perf_counter() - started) * 1000,
                "audio_base64": None,
            })
        except Exception as exc:
            logging.getLogger(__name__).exception("OpenAI live translation failed")
            socketio_emit(self.sid, "live_translation_error", {
                "error": "Không dịch được câu vừa nhận: " + str(exc)
            })
            return
        if self.tts_enabled:
            task = asyncio.create_task(self._synthesize_translation(text, vietnamese))
            self.translation_tasks.add(task)
            task.add_done_callback(self.translation_tasks.discard)

    async def _synthesize_translation(self, text, vietnamese):
        try:
            spoken = await asyncio.to_thread(self.tts_engine.synthesize, vietnamese)
            if self.tts_enabled and spoken.get("audio"):
                socketio_emit(self.sid, "live_tts_audio", {
                    "english": text,
                    "tts_ms": spoken.get("latency_ms", 0),
                    "audio_base64": base64.b64encode(spoken["audio"]).decode("ascii"),
                })
        except Exception as exc:
            logging.getLogger(__name__).exception("OpenAI live TTS failed")
            socketio_emit(self.sid, "live_tts_error", {
                "error": "Đã dịch được văn bản nhưng không tạo được giọng nói: " + str(exc)
            })

    def send_audio(self, data):
        if self.closed or not self.loop.is_running() or self.audio_queue is None:
            return False
        future = asyncio.run_coroutine_threadsafe(self.audio_queue.put(data), self.loop)
        try:
            future.result(timeout=2)
            return True
        except Exception:
            future.cancel()
            return False

    def commit_turn(self):
        if self.closed or not self.loop.is_running() or self.audio_queue is None:
            return False
        future = asyncio.run_coroutine_threadsafe(self.audio_queue.put("__commit__"), self.loop)
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
        if self.loop.is_running() and self.audio_queue is not None:
            asyncio.run_coroutine_threadsafe(self.audio_queue.put(None), self.loop)
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=8)


def socketio_emit(sid, event, payload):
    socketio.emit(event, payload, to=sid)


def configure_openai_live_socketio(socketio_instance, translator, tts_engine):
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
            emit("live_error", {"error": "Không mở được OpenAI Realtime: " + str(exc)})

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
        if len(pcm) % 2:
            emit("live_error", {"error": "Audio PCM 16-bit không hợp lệ."})
            return
        with _live_lock:
            session = _live_sessions.get(sid)
        if session and pcm and not session.send_audio(pcm):
            emit("live_error", {"error": "Hàng đợi audio đầy hoặc phiên đã dừng."})

    @socketio.on("live_commit")
    def handle_live_commit():
        sid = __import__("flask").request.sid
        with _live_lock:
            session = _live_sessions.get(sid)
        if session and not session.commit_turn():
            emit("live_error", {"error": "Không thể chốt câu audio hiện tại."})

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
