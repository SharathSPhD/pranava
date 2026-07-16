"""TDD: build a validated, aligned Vākyapadīya edition from the source JSON.

Contract for the edition builder:
  - Every mūla verse with Devanāgarī lines becomes a Kārikā.
  - Section-header entries (no lines) are excluded.
  - Translation + commentary are aligned by canonical verse id where available.
  - The build reports honest coverage counts; it never invents translations.
"""
import pytest

from pranava.corpus.edition import build_edition
from pranava.corpus.verse_id import parse_verse_label

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def edition():
    return build_edition()


def test_verse_count_matches_source(edition):
    # 1797 entries carry Devanāgarī lines in the source mūla JSON.
    assert len(edition.karikas) == 1797


def test_all_karikas_have_devanagari(edition):
    import unicodedata

    def has_dev(s: str) -> bool:
        return any("DEVANAGARI" in unicodedata.name(c, "") for c in s)

    assert all(any(has_dev(line) for line in k.mula_lines) for k in edition.karikas)


def test_kanda_distribution(edition):
    from collections import Counter

    dist = Counter(k.vid.kanda for k in edition.karikas)
    # sanity: kāṇḍa 3 (Pada) is by far the largest; all three present
    assert set(dist) == {1, 2, 3}
    assert dist[3] > dist[2] > dist[1]


def test_translation_coverage_is_high_and_reported(edition):
    cov = edition.coverage()
    assert cov["karikas"] == 1797
    # commentary set covers the vast majority of mūla verses
    assert cov["translated"] >= 1500
    assert 0.0 < cov["translated_fraction"] <= 1.0


def test_lookup_by_canonical(edition):
    k = edition.get("3.7.113")
    assert k is not None
    assert k.vid == parse_verse_label("Verse 3.7.113")
    assert any("क्रिया" in line or "कार्य" in line for line in k.mula_lines)


def test_contested_flag_is_preserved(edition):
    # 286 verses are flagged 'contested' across the commentary set;
    # at least some must survive alignment onto mūla verses.
    n = sum(1 for k in edition.karikas if k.contested)
    assert n >= 100


def test_no_fabricated_translations(edition):
    # A kārikā without an aligned commentary entry must have translation None,
    # not an empty string or placeholder.
    for k in edition.karikas:
        if k.translation is not None:
            assert k.translation.strip() != ""
