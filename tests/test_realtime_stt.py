from queue import Queue
from unittest.mock import patch

from stt.realtime_stt import handle_stt_result
from stt.sentence_buffer import SentenceBuffer


def make_result(text, error=None):
    return {
        "segment_id": 1,
        "speech_end_time": 100.0,
        "text": text,
        "latency_ms": 500.0,
        "audio_duration": 2.0,
        "error": error,
    }


def test_handle_stt_result_queues_complete_sentence():
    translation_queue = Queue()
    sentence_buffer = SentenceBuffer()

    with patch("builtins.print"):
        handle_stt_result(
            make_result("I want to go to the airport."),
            translation_queue,
            sentence_buffer,
        )

    item = translation_queue.get_nowait()

    assert item[0] == 1
    assert item[1] == "I want to go to the airport."
    assert item[2] == 500.0
    assert item[3] == 100.0
    assert sentence_buffer.buffer == ""


def test_handle_stt_result_reports_stt_error():
    translation_queue = Queue()
    sentence_buffer = SentenceBuffer()

    with patch("builtins.print") as mock_print:
        handle_stt_result(
            make_result("", error="API connection failed"),
            translation_queue,
            sentence_buffer,
        )

    assert translation_queue.empty()
    mock_print.assert_called_once()
    assert "API connection failed" in str(mock_print.call_args)


def test_handle_stt_result_flushes_incomplete_text():
    translation_queue = Queue()
    sentence_buffer = SentenceBuffer()

    with patch("builtins.print"):
        handle_stt_result(
            make_result("I want to go to the airport"),
            translation_queue,
            sentence_buffer,
        )

    item = translation_queue.get_nowait()

    assert item[1] == "I want to go to the airport"
    assert sentence_buffer.buffer == ""
