"""TDD: Devanāgarī→IAST transliteration and text integrity for the edition.

Requirements:
  - Each mūla line transliterates to IAST correctly (golden pairs).
  - Verse-number decoration (॥ १ ॥) and daṇḍa are handled, not mangled into letters.
  - Every line is valid Unicode in the Devanāgarī block (no mojibake).
  - Transliteration is deterministic.
"""
import pytest

from pranava.corpus.translit import to_iast, strip_verse_markers, devanagari_ratio

# Hand-verified golden pairs (mūla → IAST), independent of the library internals.
GOLDEN = [
    ("अनादिनिधनं ब्रह्म शब्दतत्त्वं यदक्षरम्", "anādinidhanaṃ brahma śabdatattvaṃ yadakṣaram"),
    ("विवर्ततेऽर्थभावेन", "vivartate'rthabhāvena"),
    ("एकमेव यदाम्नातं", "ekameva yadāmnātaṃ"),
]


@pytest.mark.parametrize("deva, iast", GOLDEN)
def test_golden_iast(deva, iast):
    assert to_iast(deva) == iast


def test_strip_verse_markers():
    line = "अभेदग्रहणादेष कार्यकारणयोः क्रमः ॥ ११३ ॥"
    stripped = strip_verse_markers(line)
    assert "॥" not in stripped
    assert "११३" not in stripped
    assert stripped.strip().endswith("क्रमः")


def test_daṇḍa_not_transliterated_as_letter():
    out = to_iast("क्रमः ।")
    # daṇḍa should not become a Latin letter; it maps to '|' or is dropped.
    assert "kramaḥ" in out
    assert not any(c.isascii() and c.isalpha() for c in out.replace("kramaḥ", ""))


def test_determinism():
    s = "प्रत्यस्तमितभेदं यत्"
    assert to_iast(s) == to_iast(s)


def test_devanagari_ratio_detects_mojibake():
    assert devanagari_ratio("अनादिनिधनं ब्रह्म") > 0.8
    assert devanagari_ratio("hello world") == 0.0


@pytest.mark.corpus
def test_all_edition_lines_are_valid_devanagari():
    from pranava.corpus.edition import build_edition

    ed = build_edition()
    bad = []
    for k in ed.karikas:
        for line in k.mula_lines:
            # every mūla line must be predominantly Devanāgarī (allowing markers/spaces)
            if devanagari_ratio(line) < 0.5:
                bad.append((k.canonical, line))
    assert not bad, f"{len(bad)} lines fail Devanāgarī integrity, e.g. {bad[:3]}"


@pytest.mark.corpus
def test_all_edition_lines_transliterate_without_error():
    from pranava.corpus.edition import build_edition

    ed = build_edition()
    n = 0
    for k in ed.karikas:
        for line in k.mula_lines:
            out = to_iast(line)
            assert isinstance(out, str) and out.strip()
            n += 1
    assert n > 3500  # ~2 lines per verse × 1797
