import re


class TranscriptStabilizer:
    def __init__(self, stable_words=2):
        if stable_words < 0:
            raise ValueError("stable_words must be >= 0")

        self.stable_words = stable_words
        self.committed_words = []
        self.previous_hypothesis = []
        self.latest_hypothesis = []

    @staticmethod
    def normalize_word(word):
        return re.sub(
            r"^[^\w']+|[^\w']+$",
            "",
            word.lower(),
        )

    @classmethod
    def normalize_words(cls, words):
        return [cls.normalize_word(word) for word in words]

    @staticmethod
    def clean_text(text):
        return re.sub(r"\s+", " ", text.strip())

    def find_common_prefix(self, old_words, new_words):
        old_normalized = self.normalize_words(old_words)
        new_normalized = self.normalize_words(new_words)

        count = 0
        limit = min(len(old_normalized), len(new_normalized))

        while count < limit:
            if old_normalized[count] != new_normalized[count]:
                break
            count += 1

        return count

    def update(self, hypothesis):
        hypothesis = self.clean_text(hypothesis)

        if not hypothesis:
            return {
                "full_text": " ".join(self.latest_hypothesis),
                "new_text": "",
            }

        current_words = hypothesis.split()
        self.latest_hypothesis = current_words

        # The first hypothesis establishes a baseline.
        if not self.previous_hypothesis:
            self.previous_hypothesis = current_words
            return {
                "full_text": hypothesis,
                "new_text": "",
            }

        # Commit only words shared by consecutive hypotheses,
        # keeping the last stable_words words uncommitted.
        common_count = self.find_common_prefix(
            self.previous_hypothesis,
            current_words,
        )
        commit_count = max(0, common_count - self.stable_words)

        already_committed = len(self.committed_words)
        new_text = ""

        if commit_count > already_committed:
            words_to_commit = current_words[
                already_committed:commit_count
            ]

            if words_to_commit:
                self.committed_words.extend(words_to_commit)
                new_text = " ".join(words_to_commit)

        self.previous_hypothesis = current_words

        # full_text always reflects the latest complete hypothesis.
        return {
            "full_text": hypothesis,
            "new_text": new_text,
        }

    def get_text(self):
        return " ".join(self.committed_words).strip()

    def flush(self):
        """Commit the remaining words when the utterance ends."""
        remaining_words = self.latest_hypothesis[
            len(self.committed_words):
        ]

        if remaining_words:
            self.committed_words.extend(remaining_words)

        return " ".join(remaining_words).strip()

    def reset(self):
        self.committed_words = []
        self.previous_hypothesis = []
        self.latest_hypothesis = []
