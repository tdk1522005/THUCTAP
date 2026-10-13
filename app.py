import base64
import io
import os
import subprocess
import time

import imageio_ffmpeg
import soundfile as sf
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)

# OpenAI is the default provider. Set AI_PROVIDER=gemini explicitly to use Gemini.
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").strip().lower()
if AI_PROVIDER == "gemini":
    from stt.gemini_transcriber import GeminiTranscriber
    from translation.gemini_translator import GeminiTranslationEngine
    from translation.gemini_tts import GeminiTTSEngine

    transcriber = GeminiTranscriber()
    translator = GeminiTranslationEngine()
    tts_engine = GeminiTTSEngine()
elif AI_PROVIDER == "openai":
    from stt.openai_transcriber import OpenAITranscriber
    from translation.openai_translator import OpenAITranslationEngine
    from translation.openai_tts import OpenAITTSEngine

    transcriber = OpenAITranscriber()
    translator = OpenAITranslationEngine()
    tts_engine = OpenAITTSEngine()
else:
    raise ValueError("AI_PROVIDER phải là 'gemini' hoặc 'openai'.")

print(f"[THUCTAP] AI_PROVIDER={AI_PROVIDER!r}; config_file={os.path.join(BASE_DIR, '.env')}")

app = Flask(__name__)
app.config["SECRET_KEY"] = "speech-translator-local"
socketio = SocketIO(app, async_mode="threading", max_http_buffer_size=5_000_000)

if AI_PROVIDER == "gemini":
    from stt.gemini_live_socket import configure_live_socketio
    configure_live_socketio(socketio, translator, tts_engine)
elif AI_PROVIDER == "openai":
    from stt.openai_live_socket import configure_openai_live_socketio
    configure_openai_live_socketio(socketio, translator, tts_engine)


def decode_audio(audio_bytes):
    """Giải mã audio thành WAV mono 16 kHz."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [
            ffmpeg_exe, "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0", "-f", "wav", "-ac", "1", "-ar", "16000", "pipe:1",
        ],
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if result.returncode != 0 or not result.stdout:
        raise ValueError("Không giải mã được audio.")
    audio, sample_rate = sf.read(io.BytesIO(result.stdout), dtype="float32")
    return audio, int(sample_rate)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "provider": AI_PROVIDER})


@socketio.on("connect")
def handle_connect():
    emit("connection_status", {"connected": True, "provider": AI_PROVIDER})


@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    """Xử lý một cụm audio đã được phát hiện khoảng dừng."""
    started = time.perf_counter()
    chunk_id = data.get("chunk_id") if isinstance(data, dict) else None
    try:
        if not isinstance(data, dict):
            emit("translation_error", {"error": "Dữ liệu không hợp lệ."})
            return
        audio_b64 = data.get("audio")
        tts_enabled = bool(data.get("tts_enabled", False))
        if not isinstance(audio_b64, str) or not audio_b64:
            emit("translation_error", {
                "chunk_id": chunk_id, "error": "Không có dữ liệu audio."
            })
            return

        audio_bytes = base64.b64decode(audio_b64, validate=True)
        audio, sample_rate = decode_audio(audio_bytes)
        if len(audio) < sample_rate // 2:
            emit("chunk_skipped", {"chunk_id": chunk_id})
            return

        stt_result = transcriber.transcribe(audio, sample_rate=sample_rate)
        english = stt_result.get("text", "").strip()
        if not english:
            emit("translation_result", {
                "chunk_id": chunk_id, "english": "", "vietnamese": "",
                "stt_ms": stt_result.get("latency_ms", 0),
                "translation_ms": 0, "tts_ms": 0,
                "total_ms": (time.perf_counter() - started) * 1000,
                "audio_base64": None, "message": "Không nhận diện được lời nói.",
            })
            return

        translation_result = translator.translate(english)
        vietnamese = translation_result.get("text", "").strip()
        audio_base64 = None
        tts_ms = 0
        if tts_enabled and vietnamese:
            tts_result = tts_engine.synthesize(vietnamese)
            tts_ms = tts_result.get("latency_ms", 0)
            if tts_result.get("audio"):
                audio_base64 = base64.b64encode(tts_result["audio"]).decode("ascii")

        emit("translation_result", {
            "chunk_id": chunk_id,
            "english": english,
            "vietnamese": vietnamese,
            "stt_ms": stt_result.get("latency_ms", 0),
            "translation_ms": translation_result.get("latency_ms", 0),
            "tts_ms": tts_ms,
            "total_ms": (time.perf_counter() - started) * 1000,
            "audio_base64": audio_base64,
            "message": "success",
        })
    except (ValueError, subprocess.TimeoutExpired) as exc:
        emit("translation_error", {
            "chunk_id": chunk_id, "error": str(exc) or "Không thể xử lý audio."
        })
    except Exception:
        app.logger.exception("Speech translation chunk failed")
        emit("translation_error", {
            "chunk_id": chunk_id,
            "error": "Xử lý cụm audio thất bại. Kiểm tra log backend.",
        })


if __name__ == "__main__":
    socketio.run(
        app, host="127.0.0.1", port=5000, debug=False,
        allow_unsafe_werkzeug=True,
    )
