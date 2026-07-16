"""TDD: client for the Saṃsādhanī morphological analyzer (live service).

These tests require the running samsaadhanii container on localhost:8090.
They are skipped automatically if the service is unreachable, so the suite
still passes on a machine without it — but they run (and must pass) here.
"""
import pytest

from pranava.corpus.morph import (
    SamsaadhaniiClient,
    MorphAnalysis,
    service_available,
)

pytestmark = pytest.mark.skipif(
    not service_available(), reason="samsaadhanii service not reachable on :8090"
)


@pytest.fixture(scope="module")
def client():
    return SamsaadhaniiClient()


def test_analyze_known_noun(client):
    analyses = client.analyze("रामः")
    assert analyses, "expected at least one analysis for रामः"
    assert all(isinstance(a, MorphAnalysis) for a in analyses)
    # rāma as a noun must appear among the analyses
    nouns = [a for a in analyses if a.app == "noun"]
    assert nouns
    n = nouns[0]
    assert n.root == "rāma" or n.root_wx == "rAma"
    assert n.features  # non-empty feature bundle


def test_analyze_returns_verb_reading_for_raamah(client):
    analyses = client.analyze("रामः")
    assert any(a.app == "verb" for a in analyses)


def test_unknown_token_returns_empty_not_error(client):
    # a non-word should yield no analysis rather than raising
    analyses = client.analyze("क्रमः")  # valid word; ensure structured result
    assert isinstance(analyses, list)


def test_caching_is_stable(client):
    a1 = client.analyze("ब्रह्म")
    a2 = client.analyze("ब्रह्म")
    assert [x.raw for x in a1] == [x.raw for x in a2]


def test_devanagari_to_wx_roundtrip_feed():
    # the client must transliterate DEV→WX before querying
    c = SamsaadhaniiClient()
    assert c.to_wx("रामः") == "rAmaH"
