"""
SentiKenya Preprocessing Unit Tests
=====================================
Validates the preprocessing pipeline on 24 sample posts across
English (8), Kiswahili (8), Sheng (5), and Code-Switched (3).

Run:
    python -m pytest tests/test_preprocessing.py -v
    python tests/test_preprocessing.py              # standalone
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.language_detector import KenyanLanguageDetector
from app.ml.sheng_tokenizer import ShengTokenizer
from scripts.preprocess import PreprocessingPipeline

LEXICON = "./data/sheng_lexicon.json"

# ═══════════════════════════════════════════════════════
# Test Corpus: 24 samples across all language types
# ═══════════════════════════════════════════════════════

ENGLISH_POSTS = [
    "The government needs to address the rising cost of living immediately",
    "Corruption continues to drain public resources meant for development",
    "Kenya's tech ecosystem is growing faster than ever before #DigitalKenya",
    "Healthcare workers deserve better pay and working conditions",
    "Fuel prices keep going up while salaries remain stagnant",
    "The new expressway has reduced travel time significantly in Nairobi",
    "Youth unemployment is a ticking time bomb we must address urgently",
    "M-Pesa has revolutionized how we do business across the country",
]

SWAHILI_POSTS = [
    "Serikali inafanya kazi nzuri katika ujenzi wa barabara",
    "Hali ya uchumi ni mbaya sana kwa wananchi wa kawaida",
    "Tunahitaji mabadiliko katika sekta ya elimu nchini Kenya",
    "Vijana wanahitaji kazi na fursa za kujiajiri",
    "Rushwa ni adui mkubwa wa maendeleo katika nchi yetu",
    "Walimu wetu wanastahili mshahara bora zaidi",
    "Ushuru umezidi na unaumiza wafanyabiashara wadogo",
    "Kilimo kinaweza kuwa suluhisho la ukosefu wa ajira nchini",
]

SHENG_POSTS = [
    "Siste hii economy ni tight sana hakuna pesa",
    "Manze serikali iko rada lakini watu hawajui",
    "Hii mambo ya tax ni noma bana inaumiza wasee",
    "Pesa imeisha bana hii life ni struggle tu",
    "Cheki vile Safaricom wamebadilisha game ya tech Kenya",
]

CODE_SWITCHED_POSTS = [
    "Youth tunahitaji chance ya kufanya hustle na kupay bills",
    "The government inafanya form tu they don't listen to wananchi",
    "Nairobi tech scene is growing lakini youth hawapati opportunities",
]

ALL_POSTS = (
    [(t, "en") for t in ENGLISH_POSTS]
    + [(t, "sw") for t in SWAHILI_POSTS]
    + [(t, "sh") for t in SHENG_POSTS]
    + [(t, "mixed") for t in CODE_SWITCHED_POSTS]
)

assert len(ALL_POSTS) >= 20, f"Need at least 20 samples, got {len(ALL_POSTS)}"


# ═══════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════

def test_pipeline_does_not_crash_on_any_sample():
    """Every sample must go through the full pipeline without error."""
    pipeline = PreprocessingPipeline(sheng_lexicon_path=LEXICON)
    errors = []
    for text, expected_lang in ALL_POSTS:
        try:
            result = pipeline.process(text)
            assert "cleaned_text" in result
            assert "language" in result
            assert "tokens" in result
            assert len(result["tokens"]) > 0
        except Exception as e:
            errors.append((text[:50], str(e)))
    assert not errors, f"Pipeline crashed on: {errors}"


def test_cleaned_text_has_no_urls():
    """URLs must be stripped from all outputs."""
    pipeline = PreprocessingPipeline(sheng_lexicon_path=LEXICON)
    texts_with_urls = [
        "Check this https://t.co/abc123 and share",
        "Tazama hii http://example.com/news kwenye site",
    ]
    for t in texts_with_urls:
        result = pipeline.process(t)
        assert "http" not in result["cleaned_text"]


def test_cleaned_text_has_no_mentions():
    """@mentions must be stripped."""
    pipeline = PreprocessingPipeline(sheng_lexicon_path=LEXICON)
    result = pipeline.process("Hey @governor_nrb your roads are terrible")
    assert "@" not in result["cleaned_text"]


def test_political_hashtags_preserved():
    """Political hashtags should be tagged as features."""
    pipeline = PreprocessingPipeline(sheng_lexicon_path=LEXICON)
    result = pipeline.process("We must act now #KenyaDecides #Elections2027")
    assert len(result["political_hashtags"]) >= 1


def test_language_detection_english():
    """English posts should be detected as 'en'."""
    pipeline = PreprocessingPipeline(sheng_lexicon_path=LEXICON)
    for text in ENGLISH_POSTS[:3]:
        result = pipeline.process(text)
        assert result["language"] == "en", f"Expected 'en' for: {text[:40]}, got {result['language']}"


def test_language_detection_swahili():
    """Swahili posts should be detected as 'sw'."""
    pipeline = PreprocessingPipeline(sheng_lexicon_path=LEXICON)
    for text in SWAHILI_POSTS[:3]:
        result = pipeline.process(text)
        assert result["language"] == "sw", f"Expected 'sw' for: {text[:40]}, got {result['language']}"


def test_language_detection_sheng():
    """Sheng posts should be detected as 'sh'."""
    pipeline = PreprocessingPipeline(sheng_lexicon_path=LEXICON)
    for text in SHENG_POSTS[:3]:
        result = pipeline.process(text)
        assert result["language"] == "sh", f"Expected 'sh' for: {text[:40]}, got {result['language']}"


def test_sheng_tokenizer_normalises_sms():
    """SMS abbreviations should be expanded."""
    tokenizer = ShengTokenizer(lexicon_path=LEXICON)
    tokens = tokenizer.tokenize("ur govt is bad 4 ppl")
    texts = [t.text for t in tokens]
    assert "your" in texts or "government" in texts, f"SMS not expanded: {texts}"


def test_token_count_reasonable():
    """Token counts should be between 1 and 200 for normal posts."""
    pipeline = PreprocessingPipeline(sheng_lexicon_path=LEXICON)
    for text, _ in ALL_POSTS:
        result = pipeline.process(text)
        assert 1 <= result["token_count"] <= 200, (
            f"Token count {result['token_count']} out of range for: {text[:40]}"
        )


def test_batch_processing():
    """Batch processing should return one result per input."""
    pipeline = PreprocessingPipeline(sheng_lexicon_path=LEXICON)
    texts = [t for t, _ in ALL_POSTS]
    results = pipeline.process_batch(texts)
    assert len(results) == len(ALL_POSTS)


def test_empty_and_edge_cases():
    """Edge cases should be handled gracefully."""
    pipeline = PreprocessingPipeline(sheng_lexicon_path=LEXICON)
    edge_cases = [
        "a",                                  # very short
        "😀😂🔥" * 5,                          # emoji only
        "https://t.co/xyz @user #tag",        # all metadata, no content
        "CORRUPTION!!!! IS!!!! BAD!!!!",       # excessive punctuation
        "pooooooooooaaaa sana",                # repeated chars
    ]
    for text in edge_cases:
        result = pipeline.process(text)
        assert "cleaned_text" in result  # no crash


# ═══════════════════════════════════════════════════════
# Standalone runner
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        ("Pipeline crash-free on all 24 samples", test_pipeline_does_not_crash_on_any_sample),
        ("URLs removed", test_cleaned_text_has_no_urls),
        ("Mentions removed", test_cleaned_text_has_no_mentions),
        ("Political hashtags preserved", test_political_hashtags_preserved),
        ("English detection", test_language_detection_english),
        ("Swahili detection", test_language_detection_swahili),
        ("Sheng detection", test_language_detection_sheng),
        ("SMS normalisation", test_sheng_tokenizer_normalises_sms),
        ("Token count sanity", test_token_count_reasonable),
        ("Batch processing", test_batch_processing),
        ("Edge cases", test_empty_and_edge_cases),
    ]

    print(f"\n{'='*60}")
    print(f"  SentiKenya Preprocessing Tests  ({len(ALL_POSTS)} sample posts)")
    print(f"{'='*60}\n")

    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    print(f"\n  Results: {passed}/{len(tests)} passed")
    print(f"  Sample counts: EN={len(ENGLISH_POSTS)}, SW={len(SWAHILI_POSTS)}, "
          f"SH={len(SHENG_POSTS)}, Mixed={len(CODE_SWITCHED_POSTS)}, "
          f"Total={len(ALL_POSTS)}")
    print()
