from stt.transcript_stabilizer import TranscriptStabilizer


def print_result(
    whisper_text,
    result,
):
    print()
    print(
        f"Whisper : {whisper_text}"
    )

    print(
        f"New     : {result['new_text']}"
    )

    print(
        f"Full    : {result['full_text']}"
    )


stabilizer = TranscriptStabilizer(
    stable_words=2
)


tests = [
    "you",

    "Hello everyone, today I want to",

    "Hello everyone, today I want to go to the",

    "Hello everyone, today I want to go to the airport",

    "Hello everyone, today I want to go to the airport and",

    "Hello everyone, today I want to go to the airport and I need to",

    "Hello everyone, today I want to go to the airport and I need to take a taxi.",
]


for text in tests:

    result = stabilizer.update(
        text
    )

    print_result(
        text,
        result,
    )