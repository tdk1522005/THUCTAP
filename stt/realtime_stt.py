
import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SECONDS = 0.1

MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

RMS_THRESHOLD = 0.008
SILENCE_SECONDS = 0.8
MIN_SPEECH_SECONDS = 0.4
PRE_ROLL_SECONDS = 0.2
TRAILING_AUDIO_SECONDS = 0.2


def load_whisper_model():
    print("Loading Faster-Whisper model...")
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    print("Whisper model loaded.")
    return model


def calculate_rms(audio):
    if len(audio) == 0:
        return 0.0

    audio = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(audio ** 2)))


def transcribe_audio(model, audio):
    audio = np.asarray(audio, dtype=np.float32)

    segments, info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    segments = list(segments)

    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()


def transcription_worker(model, stt_queue, result_queue):
    while True:
        item = stt_queue.get()

        if item is None:
            break

        segment_id, audio = item
        start = time.perf_counter()

        try:
            text = transcribe_audio(model, audio)
            error = None
        except Exception as exc:
            text = ""
            error = str(exc)

        result_queue.put({
            "segment_id": segment_id,
            "text": text,
            "latency_ms": (
                time.perf_counter() - start
            ) * 1000,
            "audio_duration": len(audio) / SAMPLE_RATE,
            "error": error,
        })


def main():
    model = load_whisper_model()

    # Queue 1: các block audio mới từ microphone.
    mic_queue = queue.Queue()

    # Queue 2: các đoạn audio đã chốt, chờ Whisper xử lý.
    stt_queue = queue.Queue()

    # Queue 3: kết quả Whisper trả về cho luồng chính.
    result_queue = queue.Queue()

    block_samples = int(BLOCK_SECONDS * SAMPLE_RATE)
    silence_required = int(SILENCE_SECONDS * SAMPLE_RATE)
    min_speech_samples = int(MIN_SPEECH_SECONDS * SAMPLE_RATE)
    trailing_samples = int(TRAILING_AUDIO_SECONDS * SAMPLE_RATE)

    pre_roll_count = max(
        1,
        round(PRE_ROLL_SECONDS / BLOCK_SECONDS),
    )
    pre_roll = deque(maxlen=pre_roll_count)

    worker = threading.Thread(
        target=transcription_worker,
        args=(model, stt_queue, result_queue),
        daemon=True,
    )
    worker.start()

    speech_blocks = []
    speech_samples = 0
    silence_samples = 0
    is_speaking = False

    next_segment_id = 1
    final_transcripts = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\nAudio status: {status}")

        mic_queue.put(
            indata[:, 0].astype(np.float32, copy=True)
        )

    print("\nSpeak English into your microphone.")
    print("Pause briefly after each sentence.")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=block_samples,
            callback=audio_callback,
        ):
            while True:
                # Hiển thị các kết quả đã xử lý xong.
                while True:
                    try:
                        result = result_queue.get_nowait()
                    except queue.Empty:
                        break

                    if result["error"]:
                        print(
                            f"\nSegment {result['segment_id']} "
                            f"error: {result['error']}"
                        )
                        continue

                    text = result["text"]

                    print(f"\nSegment {result['segment_id']}")
                    print(
                        f"Audio duration: "
                        f"{result['audio_duration']:.2f}s"
                    )
                    print(
                        f"STT latency: "
                        f"{result['latency_ms']:.0f} ms"
                    )
                    print(f"Whisper: {text or '(no text detected)'}")

                    if text:
                        final_transcripts.append(text)

                    print("\nTranscript so far:")
                    print(
                        " ".join(final_transcripts)
                        or "(No speech recognized)"
                    )

                try:
                    block = mic_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                rms = calculate_rms(block)

                if not is_speaking:
                    pre_roll.append(block)

                    if rms >= RMS_THRESHOLD:
                        is_speaking = True
                        speech_blocks = list(pre_roll)
                        speech_samples = sum(
                            len(part) for part in speech_blocks
                        )
                        silence_samples = 0
                        print(f"\nSpeech started (RMS: {rms:.4f})")

                else:
                    speech_blocks.append(block)
                    speech_samples += len(block)

                    if rms < RMS_THRESHOLD:
                        silence_samples += len(block)
                    else:
                        silence_samples = 0

                    if silence_samples >= silence_required:
                        if speech_samples >= min_speech_samples:
                            audio_segment = np.concatenate(
                                speech_blocks
                            ).astype(np.float32, copy=False)

                            # Giữ lại 0.2 giây âm thanh cuối.
                            end_index = max(
                                0,
                                len(audio_segment)
                                - silence_samples
                                + trailing_samples,
                            )
                            audio_segment = audio_segment[:end_index]

                            if len(audio_segment) > 0:
                                stt_queue.put((
                                    next_segment_id,
                                    audio_segment,
                                ))
                                print(
                                    f"Queued segment {next_segment_id}."
                                )
                                next_segment_id += 1

                        is_speaking = False
                        speech_blocks = []
                        speech_samples = 0
                        silence_samples = 0
                        pre_roll.clear()

    except KeyboardInterrupt:
        print("\nStopping microphone...")

    finally:
        # Gửi đoạn cuối nếu người dùng dừng khi vẫn đang nói.
        if is_speaking and speech_blocks:
            audio_segment = np.concatenate(
                speech_blocks
            ).astype(np.float32, copy=False)

            if len(audio_segment) >= min_speech_samples:
                stt_queue.put((
                    next_segment_id,
                    audio_segment,
                ))

        # Worker xử lý hết các đoạn đã xếp hàng trước khi dừng.
        stt_queue.put(None)
        worker.join()

        while not result_queue.empty():
            result = result_queue.get()

            if result["error"]:
                print(
                    f"\nSegment {result['segment_id']} "
                    f"error: {result['error']}"
                )
                continue

            print(
                f"\nSegment {result['segment_id']} "
                f"({result['latency_ms']:.0f} ms): "
                f"{result['text']}"
            )

            if result["text"]:
                final_transcripts.append(result["text"])

        print("\nFinal transcript:")
        print(
            " ".join(final_transcripts)
            or "(No speech recognized)"
        )
        print("\nStopped.")


if __name__ == "__main__":
    main()
