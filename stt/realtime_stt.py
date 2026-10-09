import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from stt.sentence_buffer import SentenceBuffer
from translation.translator import TranslationEngine


# =========================
# CONFIGURATION
# =========================

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SECONDS = 0.1

# Faster-Whisper
MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# Silence detection
RMS_THRESHOLD = 0.008
SILENCE_SECONDS = 0.8
MIN_SPEECH_SECONDS = 0.4
PRE_ROLL_SECONDS = 0.2
TRAILING_AUDIO_SECONDS = 0.2


# =========================
# STT
# =========================

def load_whisper_model():
    print("Loading Faster-Whisper model...")
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    print("Faster-Whisper loaded.")
    return model


def calculate_rms(audio):
    if len(audio) == 0:
        return 0.0
    audio = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(audio ** 2)))


def transcribe_audio(model, audio):
    audio = np.asarray(audio, dtype=np.float32)
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()


def transcription_worker(model, stt_queue, result_queue):
    while True:
        item = stt_queue.get()
        try:
            if item is None:
                return

            segment_id, audio, speech_end_time = item
            start_time = time.perf_counter()

            try:
                text = transcribe_audio(model, audio)
                error = None
            except Exception as exc:
                text = ""
                error = str(exc)

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


# =========================
# TRANSLATION
# =========================

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
                error = str(exc)

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


# =========================
# RESULT HANDLING
# =========================

def handle_stt_result(result, translation_queue, sentence_buffer):
    if result["error"]:
        print(
            f"\n[STT ERROR] Segment "
            f"{result['segment_id']}: {result['error']}"
        )
        return

    source_text = result["text"]
    print(f"\n--- Segment {result['segment_id']} ---")
    print(f"Audio duration: {result['audio_duration']:.2f}s")
    print(f"English: {source_text or '(No speech recognized)'}")
    print(f"STT latency: {result['latency_ms']:.0f} ms")

    if not source_text:
        return

    # This app transcribes a complete audio segment at a time.
    # Flush at the segment boundary so text without punctuation
    # is not delayed until a later utterance.
    first_sentence = sentence_buffer.add(source_text)
    sentences = []

    if first_sentence:
        sentences.append(first_sentence)

    sentences.extend(sentence_buffer.pop_sentences())

    remaining_text = sentence_buffer.flush()
    if remaining_text:
        sentences.append(remaining_text)

    for sentence_index, sentence in enumerate(sentences):
        translation_queue.put((
            result["segment_id"],
            sentence,
            result["latency_ms"],
            result["speech_end_time"],
        ))
        print(f"Sentence {sentence_index + 1}: {sentence}")


def handle_translation_result(result):
    if result["error"]:
        print(
            f"[TRANSLATION ERROR] Segment "
            f"{result['segment_id']}: {result['error']}"
        )
        return

    display_time = time.perf_counter()
    end_to_end_ms = (
        display_time - result["speech_end_time"]
    ) * 1000

    print(f"Vietnamese: {result['translated_text']}")
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


# =========================
# MAIN
# =========================

def main():
    whisper_model = load_whisper_model()
    translator = TranslationEngine()

    mic_queue = queue.Queue()
    stt_queue = queue.Queue()
    stt_result_queue = queue.Queue()
    translation_queue = queue.Queue()
    translation_result_queue = queue.Queue()

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
        args=(whisper_model, stt_queue, stt_result_queue),
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

    stt_thread.start()
    translation_thread.start()

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
                    handle_translation_result(result)

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

        translation_queue.put(None)
        translation_thread.join()

        while not translation_result_queue.empty():
            result = translation_result_queue.get()
            handle_translation_result(result)

        print("\nPipeline stopped.")


if __name__ == "__main__":
    main()
