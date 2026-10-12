from stt.sentence_buffer import SentenceBuffer


def test_waits_until_sentence_ends():
    buffer = SentenceBuffer()

    assert buffer.add("Hello everyone") is None
    assert buffer.add("today I am here.") == "Hello everyone today I am here."
    assert buffer.buffer == ""


def test_handles_question_and_unicode_punctuation():
    buffer = SentenceBuffer()

    assert buffer.add("Where is the station?") == "Where is the station?"
    assert buffer.add("Bạn khỏe không？") == "Bạn khỏe không？"


def test_flush_returns_remaining_text():
    buffer = SentenceBuffer()

    assert buffer.add("I need to go") is None
    assert buffer.flush() == "I need to go"
    assert buffer.flush() is None


def test_ignores_empty_chunks():
    buffer = SentenceBuffer()

    assert buffer.add("   ") is None
    assert buffer.buffer == ""


def test_reset_clears_buffer():
    buffer = SentenceBuffer()
    buffer.add("Incomplete sentence")

    buffer.reset()

    assert buffer.buffer == ""
    assert buffer.flush() is None


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

def test_stabilizer_to_sentence_buffer_integration():
    from stt.transcript_stabilizer import TranscriptStabilizer

    stabilizer = TranscriptStabilizer(stable_words=2)
    buffer = SentenceBuffer()

    hypotheses = [
        "Hello everyone, today I want to",
        "Hello everyone, today I want to go to the",
        "Hello everyone, today I want to go to the airport.",
    ]

    emitted_sentences = []

    for hypothesis in hypotheses:
        result = stabilizer.update(hypothesis)
        stable_text = result["new_text"]

        if stable_text:
            sentence = buffer.add(stable_text)
            if sentence:
                emitted_sentences.append(sentence)

    # Flush phần từ stabilizer chưa commit, rồi đưa vào buffer.
    remaining_text = stabilizer.flush()
    if remaining_text:
        sentence = buffer.add(remaining_text)
        if sentence:
            emitted_sentences.append(sentence)

    # Lấy phần cuối nếu buffer còn giữ văn bản chưa có dấu kết câu.
    remaining_sentence = buffer.flush()
    if remaining_sentence:
        emitted_sentences.append(remaining_sentence)

    assert emitted_sentences == [
        "Hello everyone, today I want to go to the airport."
    ]
