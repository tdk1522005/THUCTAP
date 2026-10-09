import re


class SentenceBuffer:
    """Gom các chunk văn bản ổn định thành câu hoàn chỉnh."""

    SENTENCE_END_PATTERN = re.compile(r"[.!?。！？]\s*$")

    def __init__(self):
        self.buffer = ""
        self.pending_sentence = None

    def has_sentence_end(self, text):
        return bool(self.SENTENCE_END_PATTERN.search(text.strip()))

    @staticmethod
    def clean_sentence(text):
        text = text.strip()
        return re.sub(r"\s+", " ", text)

    def add(self, text):
        text = self.clean_sentence(text)

        if not text:
            return None

        if self.pending_sentence is not None:
            self.buffer = self.pending_sentence + " " + text
            self.pending_sentence = None
        elif self.buffer:
            self.buffer += " " + text
        else:
            self.buffer = text

        self.buffer = self.clean_sentence(self.buffer)

        if self.has_sentence_end(self.buffer):
            sentence = self.buffer
            self.buffer = ""
            return sentence

        return None

    def flush(self):
        if self.pending_sentence is not None:
            sentence = self.pending_sentence
            self.pending_sentence = None
            return sentence

        if self.buffer:
            sentence = self.buffer
            self.buffer = ""
            return sentence

        return None

    def reset(self):
        self.buffer = ""
        self.pending_sentence = None


    def test_integrated_chunks_produce_sentence_once():
        buffer = SentenceBuffer()

        chunks = [
            "I want to",
            "go to the airport",
            "and take a taxi.",
        ]

        results = [buffer.add(chunk) for chunk in chunks]

        sentences = [result for result in results if result is not None]

        assert sentences == [
            "I want to go to the airport and take a taxi."
        ]
        assert buffer.flush() is None
