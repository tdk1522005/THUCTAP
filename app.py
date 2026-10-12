
import base64
import io
import subprocess
import time

import imageio_ffmpeg
import soundfile as sf
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit

from stt.openai_transcriber import OpenAITranscriber
from translation.openai_translator import OpenAITranslationEngine
from translation.openai_tts import OpenAITTSEngine


app = Flask(__name__)
app.config["SECRET_KEY"] = "speech-translator-local"

socketio = SocketIO(
    app,
    async_mode="threading",
    max_http_buffer_size=5_000_000,
)

# Khoi tao engine mot lan khi backend khoi dong.
transcriber = OpenAITranscriber()
translator = OpenAITranslationEngine()
tts_engine = OpenAITTSEngine()


def decode_audio(audio_bytes):
    """Giai ma audio thanh WAV mono 16 kHz."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    result = subprocess.run(
        [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "wav",
            "-ac", "1",
            "-ar", "16000",
            "pipe:1",
        ],
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )

    if result.returncode != 0 or not result.stdout:
        raise ValueError("Khong giai ma duoc audio.")

    audio, sample_rate = sf.read(
        io.BytesIO(result.stdout),
        dtype="float32",
    )
    return audio, int(sample_rate)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@socketio.on("connect")
def handle_connect():
    emit("connection_status", {"connected": True})


@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    """Xu ly mot cum audio da duoc phat hien khoang dung."""
    started = time.perf_counter()

    try:
        if not isinstance(data, dict):
            emit("translation_error", {"error": "Du lieu khong hop le."})
            return

        chunk_id = data.get("chunk_id")
        audio_b64 = data.get("audio")
        tts_enabled = bool(data.get("tts_enabled", False))

        if not isinstance(audio_b64, str) or not audio_b64:
            emit("translation_error", {
                "chunk_id": chunk_id,
                "error": "Khong co du lieu audio.",
            })
            return

        audio_bytes = base64.b64decode(audio_b64, validate=True)
        audio, sample_rate = decode_audio(audio_bytes)

        if len(audio) < sample_rate // 2:
            emit("chunk_skipped", {"chunk_id": chunk_id})
            return

        stt_result = transcriber.transcribe(
            audio,
            sample_rate=sample_rate,
        )
        english = stt_result.get("text", "").strip()

        if not english:
            emit("translation_result", {
                "chunk_id": chunk_id,
                "english": "",
                "vietnamese": "",
                "stt_ms": stt_result.get("latency_ms", 0),
                "translation_ms": 0,
                "total_ms": (time.perf_counter() - started) * 1000,
                "audio_base64": None,
                "message": "Khong nhan dien duoc loi noi.",
            })
            return

        translation_result = translator.translate(english)
        vietnamese = translation_result.get("text", "").strip()

        audio_base64 = None
        tts_ms = 0

        # Chi tao audio neu TTS dang bat khi cum duoc xu ly.
        if tts_enabled and vietnamese:
            tts_result = tts_engine.synthesize(vietnamese)
            tts_ms = tts_result.get("latency_ms", 0)
            if tts_result.get("audio"):
                audio_base64 = base64.b64encode(
                    tts_result["audio"]
                ).decode("ascii")

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
            "error": str(exc) or "Khong the xu ly audio.",
        })
    except Exception:
        app.logger.exception("Streaming chunk failed")
        emit("translation_error", {
            "error": "Xu ly cum audio that bai. Kiem tra log backend.",
        })


if __name__ == "__main__":
    socketio.run(
        app,
        host="127.0.0.1",
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True,
    )
