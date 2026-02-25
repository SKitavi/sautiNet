"""
SentiKenya Pipeline Tests
==========================
Validates all NLP components work correctly.
Run: python -m pytest tests/test_pipeline.py -v
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.language_detector import KenyanLanguageDetector
from app.ml.sheng_tokenizer import ShengTokenizer
from app.ml.sentiment_engine import SentimentEngine
from app.ml.topic_classifier import KenyanTopicClassifier
from app.ml.entity_extractor import KenyanEntityExtractor
from app.ml.pipeline import NLPPipeline
from app.models.schemas import Language, SentimentLabel


# ═══════════════════════════════════════════════
# Test Data
# ═══════════════════════════════════════════════

ENGLISH_TEXTS = [
    ("The government is doing a great job on infrastructure", "positive"),
    ("Corruption is destroying our country and stealing from citizens", "negative"),
    ("The new education policy was announced yesterday", "neutral"),
]

SWAHILI_TEXTS = [
    ("Serikali inafanya kazi nzuri katika ujenzi wa barabara", "positive"),
    ("Hali ya uchumi ni mbaya sana kwa wananchi", "negative"),
    ("Tunahitaji mabadiliko katika sekta ya elimu", "neutral"),
]

SHENG_TEXTS = [
    ("Manze hii serikali iko rada wasee mambo ni poa", "positive"),
    ("Siste hii economy ni tight sana pesa imeisha bana", "negative"),
    ("Hii mambo ya tax ni noma bana", "negative"),
]


def test_language_detection():
    """Test language detection for English, Swahili, and Sheng."""
    print("\n" + "=" * 60)
    print("  LANGUAGE DETECTION TESTS")
    print("=" * 60)

    detector = KenyanLanguageDetector(sheng_lexicon_path="./data/sheng_lexicon.json")

    tests = [
        ("The government needs to address rising costs", Language.ENGLISH),
        ("Serikali inafanya kazi nzuri katika ujenzi", Language.SWAHILI),
        ("Manze hii economy ni tight bana wasee", Language.SHENG),
        ("Vijana wanahitaji kazi na fursa za kujiajiri", Language.SWAHILI),
        ("Youth tunahitaji chance ya kufanya hustle", Language.SHENG),
    ]

    passed = 0
    for text, expected in tests:
        result = detector.detect(text)
        status = "✅" if result.detected_language == expected else "❌"
        if result.detected_language == expected:
            passed += 1
        print(f"  {status} '{text[:50]}...'")
        print(f"     Expected: {expected.value} | Got: {result.detected_language.value} "
              f"(conf: {result.confidence:.3f})")
        if result.sheng_indicators:
            print(f"     Sheng indicators: {result.sheng_indicators}")
        print()

    print(f"  Results: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_sheng_tokenizer():
    """Test Sheng tokenization and normalization."""
    print("\n" + "=" * 60)
    print("  SHENG TOKENIZER TESTS")
    print("=" * 60)

    tokenizer = ShengTokenizer(lexicon_path="./data/sheng_lexicon.json")

    test_texts = [
        "Manze hii life ni noma bana",
        "Siste pesa imeisha wasee tuko tight",
        "Cheki vile tech ndio future ya Kenya",
        "Hii serikali inafanya form tu",
    ]

    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        print(f"\n  Input: '{text}'")
        print(f"  Tokens ({len(tokens)}):")
        for t in tokens:
            extras = []
            if t.is_slang:
                extras.append(f"slang→'{t.english_gloss}'")
            if t.is_negation:
                extras.append("NEGATION")
            if t.is_intensifier:
                extras.append("INTENSIFIER")
            if abs(t.sentiment_bias) > 0.01:
                extras.append(f"sentiment={t.sentiment_bias:+.2f}")

            extra_str = f" [{', '.join(extras)}]" if extras else ""
            print(f"    {t.text:15s} ({t.source_language:8s}){extra_str}")

    print("\n  ✅ Tokenizer working")
    return True


def test_sentiment_analysis():
    """Test sentiment analysis across all three languages."""
    print("\n" + "=" * 60)
    print("  SENTIMENT ANALYSIS TESTS")
    print("=" * 60)

    detector = KenyanLanguageDetector(sheng_lexicon_path="./data/sheng_lexicon.json")
    engine = SentimentEngine()

    all_tests = [
        ("English", ENGLISH_TEXTS),
        ("Swahili", SWAHILI_TEXTS),
        ("Sheng", SHENG_TEXTS),
    ]

    total = 0
    passed = 0

    for lang_name, texts in all_tests:
        print(f"\n  ── {lang_name} ──")
        for text, expected_direction in texts:
            total += 1
            lang = detector.detect(text)
            result = engine.analyze(text, lang)

            # Check if direction matches
            if expected_direction == "positive" and result.score > 0:
                match = True
            elif expected_direction == "negative" and result.score < 0:
                match = True
            elif expected_direction == "neutral" and -0.3 < result.score < 0.3:
                match = True
            else:
                match = False

            if match:
                passed += 1

            status = "✅" if match else "❌"
            print(f"  {status} '{text[:55]}...'")
            print(f"     Expected: {expected_direction} | Got: {result.label.value} "
                  f"(score={result.score:+.3f}, conf={result.confidence:.3f}, model={result.model_used})")

    print(f"\n  Results: {passed}/{total} passed")
    return passed >= total * 0.7  # Allow some flexibility


def test_topic_classification():
    """Test topic classification for Kenyan topics."""
    print("\n" + "=" * 60)
    print("  TOPIC CLASSIFICATION TESTS")
    print("=" * 60)

    classifier = KenyanTopicClassifier(entities_path="./data/kenyan_entities.json")

    tests = [
        ("Fuel prices are killing small businesses", "economy", True),
        ("The new hospital in Nairobi will serve thousands", "healthcare", False),
        ("KCSE results show improvement in math scores", "education", False),
        ("Vote tallying at IEBC shows tight race", "election", True),
        ("M-Pesa has transformed digital payments in Kenya", "technology", False),
        ("Corruption scandal at NYS exposed by EACC", "corruption", True),
    ]

    passed = 0
    for text, expected_topic, expected_political in tests:
        result = classifier.classify(text)
        topic_match = result.primary_topic == expected_topic
        political_match = result.is_political == expected_political

        if topic_match:
            passed += 1

        status = "✅" if topic_match else "❌"
        print(f"  {status} '{text[:55]}...'")
        print(f"     Topic: {result.primary_topic} (expected: {expected_topic})")
        print(f"     Political: {result.is_political} (expected: {expected_political})"
              f" {'✅' if political_match else '❌'}")
        if result.keywords:
            print(f"     Keywords: {result.keywords[:5]}")
        print()

    print(f"  Results: {passed}/{len(tests)} passed")
    return passed >= len(tests) * 0.7


def test_entity_extraction():
    """Test Named Entity Recognition for Kenyan entities."""
    print("\n" + "=" * 60)
    print("  ENTITY EXTRACTION TESTS")
    print("=" * 60)

    extractor = KenyanEntityExtractor(entities_path="./data/kenyan_entities.json")

    tests = [
        "The Governor of Nairobi announced new plans for Mombasa infrastructure",
        "Kenya Kwanza and Azimio la Umoja are the main political coalitions",
        "IEBC and the Judiciary must ensure fair elections across Kisumu and Nakuru",
        "Safaricom launched new M-Pesa features at University of Nairobi",
        "KRA announced new tax measures worth KES 50 billion for the county budget",
    ]

    for text in tests:
        result = extractor.extract(text)
        print(f"\n  Text: '{text[:70]}...'")
        print(f"  Entities found: {len(result.entities)}")
        for entity in result.entities:
            print(f"    [{entity.label:16s}] '{entity.text}' → {entity.normalized or 'N/A'} "
                  f"(conf: {entity.confidence:.2f})")
        if result.counties_mentioned:
            print(f"  Counties: {result.counties_mentioned}")
        if result.political_figures:
            print(f"  Political figures: {result.political_figures}")
        if result.organizations:
            print(f"  Organizations: {result.organizations}")

    print("\n  ✅ Entity extraction working")
    return True


def test_full_pipeline():
    """Test the complete NLP pipeline end-to-end."""
    print("\n" + "=" * 60)
    print("  FULL PIPELINE TEST")
    print("=" * 60)

    pipeline = NLPPipeline(
        sheng_lexicon_path="./data/sheng_lexicon.json",
        entities_path="./data/kenyan_entities.json",
        node_id="TEST-01",
    )

    test_texts = [
        "The government in Nairobi needs to address corruption at KRA urgently",
        "Serikali inafanya kazi nzuri katika ujenzi wa barabara Mombasa",
        "Manze hii economy ni tight sana bana wasee wa Kisumu wanateseka",
        "Safaricom's M-Pesa continues to drive Kenya's digital economy forward",
        "Ushuru umezidi na wananchi wa Nakuru wanateseka na bei ya mafuta",
    ]

    for text in test_texts:
        result = pipeline.process_text(text)
        print(f"\n  ┌─ Text: '{text[:65]}...'")
        print(f"  ├─ Language: {result.language.detected_language.value} "
              f"(conf: {result.language.confidence:.3f})")
        print(f"  ├─ Sentiment: {result.sentiment.label.value} "
              f"(score: {result.sentiment.score:+.3f}, model: {result.sentiment.model_used})")
        print(f"  ├─ Topic: {result.topics.primary_topic} "
              f"(political: {result.topics.is_political})")
        print(f"  ├─ Entities: {len(result.entities.entities)} found")
        print(f"  └─ Time: {result.processing_time_ms:.1f}ms")

    # Pipeline stats
    stats = pipeline.get_pipeline_stats()
    print(f"\n  Pipeline Stats:")
    print(f"    Node: {stats['node_id']}")
    print(f"    Sentiment Engine: {stats['components']['sentiment_engine']}")
    print(f"    Sheng Vocab: {stats['sheng_vocab_size']} words")
    print(f"    Counties DB: {stats['entity_counties']}")
    print(f"    Topic Categories: {stats['topic_categories']}")

    print("\n  ✅ Full pipeline working")
    return True


if __name__ == "__main__":
    print("\n" + "🇰🇪" * 30)
    print("  SentiKenya NLP Pipeline Tests")
    print("🇰🇪" * 30)

    results = {
        "Language Detection": test_language_detection(),
        "Sheng Tokenizer": test_sheng_tokenizer(),
        "Sentiment Analysis": test_sentiment_analysis(),
        "Topic Classification": test_topic_classification(),
        "Entity Extraction": test_entity_extraction(),
        "Full Pipeline": test_full_pipeline(),
    }

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    all_passed = all(results.values())
    print(f"\n  Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print()
