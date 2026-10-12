from stt.transcript_stabilizer import TranscriptStabilizer


def test_full_text_tracks_latest_hypothesis():
    stabilizer = TranscriptStabilizer(stable_words=2)

    hypotheses = [
        "Hello everyone, today I want to",
        "Hello everyone, today I want to go to the",
        "Hello everyone, today I want to go to the airport",
        "Hello everyone, today I want to go to the airport and I need to take a taxi.",
    ]

    result = None
    for hypothesis in hypotheses:
        result = stabilizer.update(hypothesis)

    assert result is not None
    assert result["full_text"] == hypotheses[-1]


def test_stable_text_is_committed_without_duplicates():
    stabilizer = TranscriptStabilizer(stable_words=2)

    stabilizer.update("Hello everyone, today I want to")
    result = stabilizer.update(
        "Hello everyone, today I want to go to the"
    )

    assert result["new_text"] == "Hello everyone, today I"


def test_flush_returns_uncommitted_final_words():
    stabilizer = TranscriptStabilizer(stable_words=2)

    stabilizer.update("Hello everyone, today I want to")
    stabilizer.update("Hello everyone, today I want to go to the")

    remaining = stabilizer.flush()

    assert remaining == "want to go to the"
    assert stabilizer.get_text() == (
        "Hello everyone, today I want to go to the"
    )
