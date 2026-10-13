"""Manual local experiment; not executed during pytest collection."""


def main():
    from translation.translator import TranslationEngine
    
    
    translator = TranslationEngine()
    
    test_sentences = [
        "I want to go to the airport.",
        "Hello everyone, today I want to go to the airport.",
        "Can you tell me where the train station is?",
        "I want to book a hotel room for two people for three nights.",
        "If the flight is delayed, can we change the pickup time?"
    ]
    
    
    for i, sentence in enumerate(test_sentences, start=1):
    
        result = translator.translate(sentence)
    
        print()
        print(f"Test {i}")
        print("English   :", sentence)
        print("Vietnamese :", result["text"])
        print(f"Latency   : {result['latency_ms']:.0f} ms")


if __name__ == "__main__":
    main()
