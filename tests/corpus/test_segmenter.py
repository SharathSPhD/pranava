"""TDD: vidyut-cheda segmentation wrapper (requires the downloaded vidyut data)."""
import pytest

from pranava.corpus.segmenter import Pada, data_available, segment_line

pytestmark = pytest.mark.skipif(not data_available(), reason="vidyut data not downloaded")


def test_splits_a_known_sandhi():
    # यदक्षरम् = yat + akṣaram; the segmenter should produce >1 pāda
    padas = segment_line("यदक्षरम्")
    assert len(padas) >= 2
    assert all(isinstance(p, Pada) for p in padas)
    joined = "".join(p.text_slp for p in padas)
    assert "akzaram" in joined or "akzara" in joined


def test_returns_devanagari_and_slp():
    padas = segment_line("ब्रह्म")
    assert padas
    for p in padas:
        assert p.text_dev
        assert p.text_slp


def test_empty_line_returns_empty():
    assert segment_line("॥ १ ॥") == []


def test_multiword_line_segments():
    padas = segment_line("अनादिनिधनं ब्रह्म यदक्षरम्")
    # at least the three surface words, likely more after splitting
    assert len(padas) >= 3


def test_deterministic():
    a = segment_line("कार्यकारणयोः क्रमः")
    b = segment_line("कार्यकारणयोः क्रमः")
    assert [p.text_slp for p in a] == [p.text_slp for p in b]
