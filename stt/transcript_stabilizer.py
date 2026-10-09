import re


class TranscriptStabilizer:
    def __init__(
        self,
        stable_words=2,
    ):
        """
        stable_words:
            Số từ cuối cùng được giữ lại làm vùng
            chưa ổn định.

        Ví dụ:

            Hello everyone today I want to go

        thì phần cuối có thể tiếp tục được Whisper sửa.
        """

        self.stable_words = stable_words

        # Phần transcript đã commit.
        self.committed_text = ""

        # Hypothesis gần nhất từ Whisper.
        self.previous_hypothesis = ""

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    def normalize_word(self, word):
        word = word.lower()

        word = re.sub(
            r"^[^\w']+|[^\w']+$",
            "",
            word,
        )

        return word

    def normalize_words(self, words):
        return [
            self.normalize_word(word)
            for word in words
        ]

    def clean_text(self, text):
        text = text.strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text

    # =========================================================
    # COMMON PREFIX
    # =========================================================

    def find_common_prefix(
        self,
        old_words,
        new_words,
    ):
        """
        Tìm số từ giống nhau từ đầu.

        Ví dụ:

        old:
            Hello everyone today I want

        new:
            Hello everyone today I want to go

        => 5 từ chung
        """

        old_normalized = (
            self.normalize_words(old_words)
        )

        new_normalized = (
            self.normalize_words(new_words)
        )

        count = 0

        max_count = min(
            len(old_normalized),
            len(new_normalized),
        )

        while count < max_count:

            if (
                old_normalized[count]
                != new_normalized[count]
            ):
                break

            count += 1

        return count

    # =========================================================
    # UPDATE
    # =========================================================

    def update(self, hypothesis):
        """
        Nhận hypothesis mới từ Whisper.

        Trả về:

            full_text:
                transcript hiện tại

            new_text:
                phần text mới được commit
        """

        hypothesis = self.clean_text(
            hypothesis
        )

        if not hypothesis:
            return {
                "full_text": self.committed_text,
                "new_text": "",
            }

        # -----------------------------------------------------
        # Lần đầu
        # -----------------------------------------------------

        if not self.previous_hypothesis:

            self.previous_hypothesis = (
                hypothesis
            )

            return {
                "full_text": self.committed_text,
                "new_text": "",
            }

        # -----------------------------------------------------
        # Nếu hypothesis mới bắt đầu khác hoàn toàn
        # -----------------------------------------------------

        old_words = (
            self.previous_hypothesis.split()
        )

        new_words = (
            hypothesis.split()
        )

        common_count = (
            self.find_common_prefix(
                old_words,
                new_words,
            )
        )

        # -----------------------------------------------------
        # Xác định phần stable
        # -----------------------------------------------------

        # Không commit vài từ cuối vì Whisper vẫn có thể
        # revision chúng ở chunk tiếp theo.
        commit_count = max(
            0,
            common_count - self.stable_words,
        )

        if commit_count > 0:

            words_to_commit = (
                old_words[
                    :commit_count
                ]
            )

            commit_text = " ".join(
                words_to_commit
            )

            # -------------------------------------------------
            # Tránh commit duplicate
            # -------------------------------------------------

            committed_words = (
                self.committed_text.split()
            )

            committed_normalized = (
                self.normalize_words(
                    committed_words
                )
            )

            commit_normalized = (
                self.normalize_words(
                    words_to_commit
                )
            )

            new_words_to_commit = []

            start_index = 0

            while (
                start_index
                < len(commit_normalized)
                and start_index
                < len(committed_normalized)
                and commit_normalized[
                    start_index
                ]
                == committed_normalized[
                    start_index
                ]
            ):
                start_index += 1

            if start_index < len(
                words_to_commit
            ):
                new_words_to_commit = (
                    words_to_commit[
                        start_index:
                    ]
                )

            if new_words_to_commit:

                new_text = " ".join(
                    new_words_to_commit
                )

                if self.committed_text:
                    self.committed_text += (
                        " "
                        + new_text
                    )
                else:
                    self.committed_text = (
                        new_text
                    )

            else:
                new_text = ""

        else:
            new_text = ""

        # -----------------------------------------------------
        # Save latest hypothesis
        # -----------------------------------------------------

        self.previous_hypothesis = (
            hypothesis
        )

        # -----------------------------------------------------
        # Full text
        #
        # committed_text + phần hypothesis hiện tại
        # -----------------------------------------------------

        committed_words = (
            self.committed_text.split()
        )

        hypothesis_words = (
            hypothesis.split()
        )

        committed_normalized = (
            self.normalize_words(
                committed_words
            )
        )

        hypothesis_normalized = (
            self.normalize_words(
                hypothesis_words
            )
        )

        # Tìm phần hypothesis chưa nằm trong committed.
        remaining_index = 0

        while (
            remaining_index
            < len(committed_normalized)
            and remaining_index
            < len(hypothesis_normalized)
            and committed_normalized[
                remaining_index
            ]
            == hypothesis_normalized[
                remaining_index
            ]
        ):
            remaining_index += 1

        remaining_words = (
            hypothesis_words[
                remaining_index:
            ]
        )

        remaining_text = " ".join(
            remaining_words
        )

        if self.committed_text:
            if remaining_text:
                full_text = (
                    self.committed_text
                    + " "
                    + remaining_text
                )
            else:
                full_text = (
                    self.committed_text
                )
        else:
            full_text = remaining_text

        return {
            "full_text": full_text.strip(),
            "new_text": new_text.strip(),
        }

    # =========================================================
    # GET TEXT
    # =========================================================

    def get_text(self):
        return self.committed_text.strip()

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):
        self.committed_text = ""
        self.previous_hypothesis = ""