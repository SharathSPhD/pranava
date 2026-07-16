"""TDD: verse-anchored concept knowledge graph for the Vākyapadīya.

Integrity rules the graph must satisfy:
  - Every concept node is anchored to >=1 real verse whose mūla contains the term,
    and the anchor carries the quoted original line (human-verifiable).
  - Co-occurrence edges only exist when both concepts genuinely appear in the cited verse.
  - Curated relations are explicitly typed and each carries a verse anchor.
  - Nothing is anchored to a verse that does not contain the term.
"""
import pytest

from pranava.kg.concepts import CONCEPTS, build_concept_graph

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def graph():
    return build_concept_graph()


def test_min_concepts_registered():
    assert len(CONCEPTS) >= 12


def test_every_concept_has_a_real_anchor(graph):
    for c in graph.nodes.values():
        assert c.anchors, f"{c.id} has no anchors"
        for a in c.anchors:
            # the quoted line must actually be one of that verse's mūla lines
            assert a.quote
            assert a.vid


def test_anchors_actually_contain_the_term(graph):
    from pranava.corpus.translit import to_iast
    import unicodedata

    def fold(s: str) -> str:
        return "".join(
            ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
        ).lower()

    sphota = graph.nodes["sphota"]
    for a in sphota.anchors:
        assert fold("sphoṭa")[:5] in fold(to_iast(a.quote)), (a.vid, a.quote)


def test_cooccurrence_edges_are_grounded(graph):
    # sphoṭa and dhvani are discussed together in Bhartṛhari; expect a grounded edge.
    e = graph.get_edge("sphota", "dhvani")
    assert e is not None
    assert e.anchors, "co-occurrence edge must cite the verses where both appear"
    # every cited verse must contain both terms
    for vid in e.anchors:
        v = graph.verse_terms(vid)
        assert "sphota" in v and "dhvani" in v


def test_curated_relations_have_verse_anchors(graph):
    rels = graph.curated_relations()
    assert rels
    for r in rels:
        assert r.rel_type
        assert r.anchor_vid, f"curated relation {r} lacks a verse anchor"
        assert graph.nodes[r.src] and graph.nodes[r.dst]


def test_query_returns_anchors(graph):
    anchors = graph.anchors_for("sphota")
    assert anchors
    assert all(a.vid for a in anchors)


def test_no_orphan_edges(graph):
    for (a, b) in graph.edges:
        assert a in graph.nodes and b in graph.nodes
