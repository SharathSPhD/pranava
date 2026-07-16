"""TDD: canonical parsing of Vākyapadīya verse labels.

The source labels come in several shapes we must handle without loss:
  - "Verse 1.5"          flat numbering (Brahma-, Vākya-kāṇḍa)
  - "Verse 3.7.113"      kāṇḍa 3 has samuddeśa sub-sections (kāṇḍa.section.verse)
  - "Verse 1.16-17"      a range spanning multiple kārikās printed as one unit
  - "Verse 2.127cd"      a half-verse (pāda c/d only)
"""
import pytest

from pranava.corpus.verse_id import VerseId, parse_verse_label


def test_flat_label():
    vid = parse_verse_label("Verse 1.5")
    assert vid.kanda == 1
    assert vid.section is None
    assert vid.start == 5
    assert vid.end == 5
    assert not vid.is_range
    assert vid.half is None
    assert vid.canonical == "1.5"


def test_sectioned_label():
    vid = parse_verse_label("Verse 3.7.113")
    assert vid.kanda == 3
    assert vid.section == 7
    assert vid.start == 113
    assert vid.canonical == "3.7.113"


def test_range_label():
    vid = parse_verse_label("Verse 1.16-17")
    assert vid.kanda == 1
    assert vid.start == 16
    assert vid.end == 17
    assert vid.is_range
    assert vid.canonical == "1.16-17"


def test_half_verse_label():
    vid = parse_verse_label("Verse 2.127cd")
    assert vid.kanda == 2
    assert vid.start == 127
    assert vid.half == "cd"
    assert vid.canonical == "2.127cd"


def test_parenthesized_half_verse():
    vid = parse_verse_label("Verse 2.307(ab)")
    assert vid.kanda == 2
    assert vid.start == 307
    assert vid.half == "ab"
    assert vid.canonical == "2.307ab"


def test_single_pada_half():
    vid = parse_verse_label("Verse 2.326a")
    assert vid.kanda == 2
    assert vid.start == 326
    assert vid.half == "a"
    assert vid.canonical == "2.326a"


def test_section_header_label_rejected():
    # "Book 1 - Brahma-kāṇḍa ..." is not a verse
    with pytest.raises(ValueError):
        parse_verse_label("Book 1 - Brahma-kāṇḍa (or Āgama-samuccaya)")


def test_ordering_is_stable_and_sanskrit_correct():
    labels = ["Verse 3.1.2", "Verse 3.1.10", "Verse 1.5", "Verse 2.1", "Verse 3.1.1"]
    ids = [parse_verse_label(x) for x in labels]
    ordered = [v.canonical for v in sorted(ids)]
    assert ordered == ["1.5", "2.1", "3.1.1", "3.1.2", "3.1.10"]


def test_equality_and_hash():
    a = parse_verse_label("Verse 3.7.113")
    b = parse_verse_label("Verse 3.7.113")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
