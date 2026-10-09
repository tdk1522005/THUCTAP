import queue
import re
import sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from translation.translator import TranslationEngine


class RealtimeSTT:
    def __init__(
        self,
        model_size="small",
        sample_rate=16000,
        channels=1,
        chunk_duration=2,
        overlap_duration=1,
        volume_threshold=0.01,
        candidate_confirmations=2,
    ):
        self.sample_rate = sample_rate
        self.channels = channels

        self.chunk_duration = chunk_duration
        self.chunk_samples = (
            sample_rate * chunk_duration
        )

        self.overlap_duration = overlap_duration
        self.overlap_samples = (
            sample_rate * overlap_duration
        )

        self.volume_threshold = volume_threshold

        # Số lần candidate phải giữ nguyên
        # trước khi dịch.
        self.candidate_confirmations = (
            candidate_confirmations
        )

        self.audio_queue = queue.Queue()

        # -----------------------------------------------------
        # Whisper
        # -----------------------------------------------------

        print("Loading Whisper model...")

        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

        print("Whisper model loaded.")

        # -----------------------------------------------------
        # Translation
        # -----------------------------------------------------

        print("Loading translation model...")

        self.translator = TranslationEngine(
            model_name="Helsinki-NLP/opus-mt-en-vi"
        )

        print("Translation model loaded.")

        # -----------------------------------------------------
        # Audio buffers
        # -----------------------------------------------------

        self.audio_buffer = np.empty(
            (0, channels),
            dtype=np.float32,
        )

        self.previous_audio = np.empty(
            0,
            dtype=np.float32,
        )

        # -----------------------------------------------------
        # Sentence state
        # -----------------------------------------------------

        self.candidate_sentence = ""
        self.candidate_count = 0

        self.translated_sentences = []

    # =========================================================
    # AUDIO CALLBACK
    # =========================================================

    def audio_callback(
        self,
        indata,
        frames,
        time,
        status,
    ):
        if status:
            print(
                f"Audio status: {status}",
                file=sys.stderr,
            )

        self.audio_queue.put(
            indata.copy()
        )

    # =========================================================
    # WHISPER
    # =========================================================

    def transcribe_chunk(self, audio):
        segments, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=5,
        )

        text = ""

        for segment in segments:
            text += segment.text

        return text.strip()

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    def normalize_text(self, text):
        text = text.lower().strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = re.sub(
            r"\s+([.!?,])",
            r"\1",
            text,
        )

        return text

    # =========================================================
    # SENTENCE DETECTION
    # =========================================================

    def extract_complete_sentences(
        self,
        text,
    ):
        """
        Tìm các câu có dấu kết thúc.

        Ví dụ:

            Hello everyone. How are you?

        sẽ trả về:

            [
                "Hello everyone.",
                "How are you?"
            ]
        """

        text = text.strip()

        if not text:
            return []

        pattern = (
            r"[^.!?。！？]+[.!?。！？]"
        )

        matches = re.findall(
            pattern,
            text,
        )

        sentences = []

        for match in matches:

            sentence = match.strip()

            if sentence:
                sentences.append(
                    sentence
                )

        return sentences

    # =========================================================
    # ALREADY TRANSLATED CHECK
    # =========================================================

    def is_already_translated(
        self,
        sentence,
    ):
        normalized = (
            self.normalize_text(
                sentence
            )
        )

        for translated in (
            self.translated_sentences
        ):
            if (
                self.normalize_text(
                    translated
                )
                == normalized
            ):
                return True

        return False

    # =========================================================
    # FIND CANDIDATE
    # =========================================================

    def find_sentence_candidate(
        self,
        whisper_text,
    ):
        sentences = (
            self.extract_complete_sentences(
                whisper_text
            )
        )

        if not sentences:
            return None

        for sentence in sentences:

            if not self.is_already_translated(
                sentence
            ):
                return sentence

        return None

    # =========================================================
    # UPDATE CANDIDATE
    # =========================================================

    def update_candidate(
        self,
        whisper_text,
    ):
        """
        Candidate phải xuất hiện ổn định nhiều lần
        trước khi được đưa sang Translation.
        """

        candidate = (
            self.find_sentence_candidate(
                whisper_text
            )
        )

        # -----------------------------------------------------
        # Không có complete sentence
        # -----------------------------------------------------

        if candidate is None:

            if self.candidate_sentence:
                print(
                    "Candidate disappeared/revised."
                )

            self.candidate_sentence = ""
            self.candidate_count = 0

            return None

        normalized_candidate = (
            self.normalize_text(
                candidate
            )
        )

        normalized_previous = (
            self.normalize_text(
                self.candidate_sentence
            )
        )

        # -----------------------------------------------------
        # Candidate giống lần trước
        # -----------------------------------------------------

        if (
            self.candidate_sentence
            and normalized_candidate
            == normalized_previous
        ):

            self.candidate_count += 1

        # -----------------------------------------------------
        # Candidate mới
        # -----------------------------------------------------

        else:

            self.candidate_sentence = (
                candidate
            )

            self.candidate_count = 1

        print(
            f"Candidate: "
            f"{self.candidate_sentence}"
        )

        print(
            "Confirmation: "
            f"{self.candidate_count}/"
            f"{self.candidate_confirmations}"
        )

        # -----------------------------------------------------
        # Chưa đủ confirmation
        # -----------------------------------------------------

        if (
            self.candidate_count
            < self.candidate_confirmations
        ):

            return None

        # -----------------------------------------------------
        # FINAL
        # -----------------------------------------------------

        final_sentence = (
            self.candidate_sentence
        )

        self.candidate_sentence = ""
        self.candidate_count = 0

        return final_sentence

    # =========================================================
    # TRANSLATION
    # =========================================================

    def translate_sentence(
        self,
        sentence,
    ):
        print()
        print(
            "======================================"
        )

        print(
            f"FINAL EN : {sentence}"
        )

        result = self.translator.translate(
            sentence
        )

        print(
            f"FINAL VI : {result['text']}"
        )

        print(
            "Translation latency: "
            f"{result['latency_ms']:.0f} ms"
        )

        print(
            "======================================"
        )
        print()

        self.translated_sentences.append(
            sentence
        )

    # =========================================================
    # PROCESS WHISPER RESULT
    # =========================================================

    def process_transcript(
        self,
        whisper_text,
    ):
        if not whisper_text:
            return

        print()
        print(
            f"Whisper : {whisper_text}"
        )

        # -----------------------------------------------------
        # Không reconstruct transcript nữa.
        #
        # Mỗi Whisper result được xem là một current
        # hypothesis.
        # -----------------------------------------------------

        final_sentence = (
            self.update_candidate(
                whisper_text
            )
        )

        if final_sentence is None:

            print(
                "Waiting for stable sentence..."
            )

            return

        self.translate_sentence(
            final_sentence
        )

    # =========================================================
    # LISTEN
    # =========================================================

    def listen(self):
        print()
        print(
            "Speak English into your microphone."
        )

        print(
            "Press Ctrl+C to stop."
        )

        print()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self.audio_callback,
        ):

            try:

                while True:

                    # -------------------------------------------------
                    # Get audio
                    # -------------------------------------------------

                    audio_chunk = (
                        self.audio_queue.get()
                    )

                    self.audio_buffer = (
                        np.concatenate(
                            (
                                self.audio_buffer,
                                audio_chunk,
                            ),
                            axis=0,
                        )
                    )

                    # -------------------------------------------------
                    # Chưa đủ 2 giây
                    # -------------------------------------------------

                    if (
                        len(self.audio_buffer)
                        < self.chunk_samples
                    ):
                        continue

                    # -------------------------------------------------
                    # Lấy chunk
                    # -------------------------------------------------

                    new_chunk = (
                        self.audio_buffer[
                            : self.chunk_samples
                        ]
                    )

                    self.audio_buffer = (
                        self.audio_buffer[
                            self.chunk_samples:
                        ]
                    )

                    new_audio = (
                        new_chunk[:, 0]
                    )

                    # -------------------------------------------------
                    # Volume
                    # -------------------------------------------------

                    volume = np.sqrt(
                        np.mean(
                            new_audio ** 2
                        )
                    )

                    print()
                    print(
                        f"Volume: {volume:.6f}"
                    )

                    if (
                        volume
                        < self.volume_threshold
                    ):

                        print(
                            "No speech detected, "
                            "skipping..."
                        )

                        continue

                    # -------------------------------------------------
                    # Add overlap
                    # -------------------------------------------------

                    if len(
                        self.previous_audio
                    ) > 0:

                        audio_for_stt = (
                            np.concatenate(
                                (
                                    self.previous_audio,
                                    new_audio,
                                )
                            )
                        )

                    else:

                        audio_for_stt = (
                            new_audio
                        )

                    print(
                        "[Transcribing "
                        f"{len(audio_for_stt) / self.sample_rate:.1f}s "
                        "audio...]"
                    )

                    # -------------------------------------------------
                    # Whisper
                    # -------------------------------------------------

                    text = (
                        self.transcribe_chunk(
                            audio_for_stt
                        )
                    )

                    # -------------------------------------------------
                    # Translation pipeline
                    # -------------------------------------------------

                    if text:

                        self.process_transcript(
                            text
                        )

                    # -------------------------------------------------
                    # Save overlap
                    # -------------------------------------------------

                    if (
                        self.overlap_samples
                        > 0
                    ):

                        self.previous_audio = (
                            new_audio[
                                -self.overlap_samples:
                            ]
                        )

                    else:

                        self.previous_audio = (
                            np.empty(
                                0,
                                dtype=np.float32,
                            )
                        )

            except KeyboardInterrupt:

                print()
                print(
                    "Stopping..."
                )


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    stt = RealtimeSTT(
        model_size="small",
        chunk_duration=2,
        overlap_duration=1,
        volume_threshold=0.01,
        candidate_confirmations=2,
    )

    stt.listen()