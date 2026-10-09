import re


class SentenceBuffer:
    def __init__(self):
        self.buffer = ""
        self.pending_sentence = None

    def has_sentence_end(self, text):
        """
        Kiểm tra text có dấu kết thúc câu.
        """

        return bool(
            re.search(
                r"[.!?。！？]\s*$",
                text.strip(),
            )
        )

    def clean_sentence(self, text):
        """
        Chuẩn hóa khoảng trắng và giữ nội dung câu.
        """

        text = text.strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text

    def add(self, text):
        """
        Nhận text mới từ Transcript Stabilizer.

        Trả về:
            sentence: câu hoàn chỉnh
            None: chưa hoàn chỉnh
        """

        text = self.clean_sentence(text)

        if not text:
            return None

        # ==================================================
        # Nếu đang có pending sentence
        # ==================================================

        if self.pending_sentence is not None:

            # Chunk mới nối tiếp câu trước.
            #
            # Ví dụ:
            #
            # pending:
            # Hello everyone, today I want to go to the airport
            #
            # text:
            # and I need to take a taxi.
            #
            # => ghép lại.
            self.buffer = (
                self.pending_sentence
                + " "
                + text
            )

            self.pending_sentence = None

        else:

            # Thêm text mới vào buffer
            if self.buffer:
                self.buffer += " " + text
            else:
                self.buffer = text

        self.buffer = self.clean_sentence(
            self.buffer
        )

        # ==================================================
        # Kiểm tra câu đã kết thúc chưa
        # ==================================================

        if self.has_sentence_end(
            self.buffer
        ):

            sentence = self.buffer

            self.buffer = ""

            return sentence

        return None

    def flush(self):
        """
        Lấy phần text còn lại khi stop.
        """

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