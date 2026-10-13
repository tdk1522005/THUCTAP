import io

from translation.openai_tts import OpenAITTSEngine
import queue
import threading
import time
from collections import deque

import numpy as np
import soundfile as sf
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from stt.openai_transcriber import OpenAITranscriber
from stt.sentence_buffer import SentenceBuffer
from translation.openai_translator import OpenAITranslationEngine


SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SECONDS = 0.1

RMS_THRESHOLD = 0.008
SILENCE_SECONDS = 0.5
MIN_SPEECH_SECONDS = 0.4
PRE_ROLL_SECONDS = 0.2
TRAILING_AUDIO_SECONDS = 0.2


def calculate_rms(audio):
    if len(audio) == 0:
        return 0.0

    audio = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(audio ** 2)))


def format_api_error(exc):
    """Return a readable message for common OpenAI API errors."""
    if isinstance(exc, AuthenticationError):
        return "API key không hợp lệ hoặc không có quyền truy cập."

    if isinstance(exc, RateLimitError):
        return "API vượt giới hạn tốc độ hoặc hạn mức tài khoản."

    if isinstance(exc, APITimeoutError):
        return "Yêu cầu API hết thời gian chờ."

    if isinstance(exc, APIConnectionError):
        return "Không thể kết nối OpenAI API. Kiểm tra mạng."

    if isinstance(exc, BadRequestError):
        return f"Yêu cầu API không hợp lệ: {exc}"

    if isinstance(exc, APIStatusError):
        return f"OpenAI API trả về HTTP {exc.status_code}: {exc}"

    return f"{type(exc).__name__}: {exc}"


def transcribe_audio(transcriber, audio):
    result = transcriber.transcribe(
        audio,
        sample_rate=SAMPLE_RATE,
    )
    return result["text"]


def transcription_worker(transcriber, stt_queue, result_queue):
    while True:
        item = stt_queue.get()

        try:
            if item is None:
                return

            segment_id, audio, speech_end_time = item
            start_time = time.perf_counter()

            try:
                text = transcribe_audio(transcriber, audio)
                error = None
            except Exception as exc:
                text = ""
                error = format_api_error(exc)

            result_queue.put({
                "segment_id": segment_id,
                "speech_end_time": speech_end_time,
                "text": text,
                "latency_ms": (
                    time.perf_counter() - start_time
                ) * 1000,
                "audio_duration": len(audio) / SAMPLE_RATE,
                "error": error,
            })

        finally:
            stt_queue.task_done()


def translation_worker(
    translator,
    translation_queue,
    translation_result_queue,
):
    while True:
        item = translation_queue.get()

        try:
            if item is None:
                return

            (
                segment_id,
                source_text,
                stt_latency_ms,
                speech_end_time,
            ) = item

            try:
                result = translator.translate(source_text)
                translated_text = result["text"]
                translation_latency_ms = result["latency_ms"]
                error = None
            except Exception as exc:
                translated_text = ""
                translation_latency_ms = 0
                error = format_api_error(exc)

            translation_result_queue.put({
                "segment_id": segment_id,
                "source_text": source_text,
                "translated_text": translated_text,
                "stt_latency_ms": stt_latency_ms,
                "translation_latency_ms": translation_latency_ms,
                "speech_end_time": speech_end_time,
                "error": error,
            })

        finally:
            translation_queue.task_done()



def tts_worker(tts_engine, tts_queue):
    """Synthesize and play Vietnamese audio sequentially."""
    while True:
        item = tts_queue.get()

        try:
            if item is None:
                return

            segment_id, text = item

            try:
                result = tts_engine.synthesize(text)
                audio_bytes = result["audio"]

                if not audio_bytes:
                    continue

                with io.BytesIO(audio_bytes) as audio_buffer:
                    audio_data, sample_rate = sf.read(
                        audio_buffer,
                        dtype="float32",
                    )

                print(
                    f"[TTS] Segment {segment_id}: "
                    f"{result['latency_ms']:.0f} ms synthesis"
                )

                import sounddevice as sd

                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()

            except Exception as exc:
                print(
                    f"[TTS ERROR] Segment {segment_id}: "
                    f"{format_api_error(exc)}"
                )

        finally:
            tts_queue.task_done()



def handle_stt_result(result, translation_queue, sentence_buffer):
    if result["error"]:
        print(
            f"\n[STT ERROR] Segment {result['segment_id']}: "
            f"{result['error']}"
        )
        return

    source_text = result["text"]
    print(f"\n--- Segment {result['segment_id']} ---")
    print(f"Audio duration: {result['audio_duration']:.2f}s")
    print(f"English: {source_text or '(No speech recognized)'}")
    print(f"STT latency: {result['latency_ms']:.0f} ms")

    if not source_text:
        return

    first_sentence = sentence_buffer.add(source_text)
    sentences = []

    if first_sentence:
        sentences.append(first_sentence)

    sentences.extend(sentence_buffer.pop_sentences())

    # Giu hanh vi cu: chuyen phan van ban con lai di dich ngay.
    remaining_text = sentence_buffer.flush()
    if remaining_text:
        sentences.append(remaining_text)

    for sentence_index, sentence in enumerate(sentences):
        print(f"Sentence {sentence_index + 1}: {sentence}")
        translation_queue.put((
            result["segment_id"],
            sentence,
            result["latency_ms"],
            result["speech_end_time"],
        ))


def handle_translation_result(result, tts_queue=None):
    if result["error"]:
        print(
            f"[TRANSLATION ERROR] Segment {result['segment_id']}: "
            f"{result['error']}"
        )
        return

    display_time = time.perf_counter()
    end_to_end_ms = (
        display_time - result["speech_end_time"]
    ) * 1000

    translated_text = result["translated_text"].strip()
    print(f"Vietnamese: {translated_text}")

    if tts_queue is not None and translated_text:
        tts_queue.put((result["segment_id"], translated_text))
    print(f"STT latency: {result['stt_latency_ms']:.0f} ms")
    print(
        "Translation latency: "
        f"{result['translation_latency_ms']:.0f} ms"
    )
    print(
        "End-to-end latency "
        "(last detected voice -> display): "
        f"{end_to_end_ms:.0f} ms"
    )
    print("-" * 50)


def main():
    print("Initializing OpenAI STT and translation...")

    try:
        transcriber = OpenAITranscriber()
        translator = OpenAITranslationEngine()
        tts_engine = OpenAITTSEngine()
    except Exception as exc:
        print(f"[INITIALIZATION ERROR] {format_api_error(exc)}")
        return

    print(f"STT model: {transcriber.model}")
    print(f"Translation model: {translator.model}")

    mic_queue = queue.Queue()
    stt_queue = queue.Queue()
    stt_result_queue = queue.Queue()
    translation_queue = queue.Queue()
    translation_result_queue = queue.Queue()
    tts_queue = queue.Queue()

    sentence_buffer = SentenceBuffer()

    block_samples = int(BLOCK_SECONDS * SAMPLE_RATE)
    silence_required = int(SILENCE_SECONDS * SAMPLE_RATE)
    min_speech_samples = int(MIN_SPEECH_SECONDS * SAMPLE_RATE)
    trailing_samples = int(TRAILING_AUDIO_SECONDS * SAMPLE_RATE)

    pre_roll_count = max(
        1,
        round(PRE_ROLL_SECONDS / BLOCK_SECONDS),
    )
    pre_roll = deque(maxlen=pre_roll_count)

    stt_thread = threading.Thread(
        target=transcription_worker,
        args=(transcriber, stt_queue, stt_result_queue),
        daemon=True,
    )
    translation_thread = threading.Thread(
        target=translation_worker,
        args=(
            translator,
            translation_queue,
            translation_result_queue,
        ),
        daemon=True,
    )

    tts_thread = threading.Thread(
        target=tts_worker,
        args=(tts_engine, tts_queue),
        daemon=True,
    )

    stt_thread.start()
    translation_thread.start()
    tts_thread.start()

    speech_blocks = []
    speech_samples = 0
    silence_samples = 0
    is_speaking = False
    last_voice_time = None
    next_segment_id = 1

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\nAudio status: {status}")
        mic_queue.put(indata[:, 0].astype(np.float32, copy=True))

    print("\nRealtime English -> Vietnamese translation")
    print("Speak English into your microphone.")
    print("Pause briefly between sentences.")
    print("Press Ctrl+C to stop.\n")

    try:
        import sounddevice as sd

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=block_samples,
            callback=audio_callback,
        ):
            while True:
                while True:
                    try:
                        result = stt_result_queue.get_nowait()
                    except queue.Empty:
                        break

                    handle_stt_result(
                        result,
                        translation_queue,
                        sentence_buffer,
                    )

                while True:
                    try:
                        result = translation_result_queue.get_nowait()
                    except queue.Empty:
                        break

                    handle_translation_result(result, tts_queue)

                try:
                    block = mic_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                rms = calculate_rms(block)

                if not is_speaking:
                    pre_roll.append(block)

                    if rms >= RMS_THRESHOLD:
                        is_speaking = True
                        last_voice_time = time.perf_counter()
                        speech_blocks = list(pre_roll)
                        speech_samples = sum(
                            len(part) for part in speech_blocks
                        )
                        silence_samples = 0
                        print("\nSpeech detected...")

                else:
                    speech_blocks.append(block)
                    speech_samples += len(block)

                    if rms < RMS_THRESHOLD:
                        silence_samples += len(block)
                    else:
                        silence_samples = 0
                        last_voice_time = time.perf_counter()

                    if silence_samples >= silence_required:
                        speech_end_time = (
                            last_voice_time or time.perf_counter()
                        )

                        if speech_samples >= min_speech_samples:
                            audio_segment = np.concatenate(
                                speech_blocks
                            ).astype(np.float32, copy=False)

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
                                    speech_end_time,
                                ))
                                print(
                                    f"Queued segment {next_segment_id}."
                                )
                                next_segment_id += 1

                        is_speaking = False
                        speech_blocks = []
                        speech_samples = 0
                        silence_samples = 0
                        last_voice_time = None
                        pre_roll.clear()

    except KeyboardInterrupt:
        print("\nStopping microphone...")

    finally:
        if is_speaking and speech_blocks:
            audio_segment = np.concatenate(
                speech_blocks
            ).astype(np.float32, copy=False)

            if len(audio_segment) >= min_speech_samples:
                stt_queue.put((
                    next_segment_id,
                    audio_segment,
                    last_voice_time or time.perf_counter(),
                ))

        stt_queue.put(None)
        stt_thread.join()

        while not stt_result_queue.empty():
            result = stt_result_queue.get()
            handle_stt_result(
                result,
                translation_queue,
                sentence_buffer,
            )
        remaining_text = sentence_buffer.flush()
        if remaining_text:
            translation_queue.put((
                next_segment_id - 1,
                remaining_text,
                0.0,
                time.perf_counter(),
            ))
        translation_queue.put(None)
        translation_thread.join()

        while not translation_result_queue.empty():
            result = translation_result_queue.get()
            handle_translation_result(result, tts_queue)

        # Cho TTS doc het cac cau dang cho roi moi ket thuc.
        tts_queue.put(None)
        tts_thread.join()

        print("\nPipeline stopped.")


if __name__ == "__main__":
    main()
