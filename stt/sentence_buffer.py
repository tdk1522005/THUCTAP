import re
from collections import deque


class SentenceBuffer:
    """Collect text chunks and emit complete sentences."""

    SENTENCE_END_PATTERN = re.compile(r'[.!?。！？]+["\u0027\u201d\u2019)]*\s*')

    def __init__(self):
        self.buffer = ""
        self.pending_sentences = deque()

    @staticmethod
    def clean_sentence(text):
        return re.sub(r"\s+", " ", text.strip())

    def add(self, text):
        text = self.clean_sentence(text)
        if not text:
            return self.pending_sentences.popleft() if self.pending_sentences else None

        self.buffer = self.clean_sentence(
            f"{self.buffer} {text}" if self.buffer else text
        )

        start = 0
        for match in self.SENTENCE_END_PATTERN.finditer(self.buffer):
            sentence = self.buffer[start:match.end()].strip()
            if sentence:
                self.pending_sentences.append(sentence)
            start = match.end()

        self.buffer = self.buffer[start:].strip()

        return (
            self.pending_sentences.popleft()
            if self.pending_sentences
            else None
        )

    def pop_sentences(self):
        """Return all complete sentences waiting in the internal queue."""
        sentences = list(self.pending_sentences)
        self.pending_sentences.clear()
        return sentences

    def flush(self):
        """Return remaining text, including any queued complete sentences."""
        if self.pending_sentences:
            sentences = list(self.pending_sentences)
            self.pending_sentences.clear()
            if self.buffer:
                sentences.append(self.buffer)
                self.buffer = ""
            return " ".join(sentences)

        remaining = self.clean_sentence(self.buffer)
        self.buffer = ""
        return remaining or None

    def reset(self):
        self.buffer = ""
        self.pending_sentences.clear()
