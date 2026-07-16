"""Verse-anchored concept knowledge graph for Bhartṛhari's Vākyapadīya.

Design for scholarly integrity:
  * **Nodes** are technical concepts. Each is *empirically anchored*: we search the
    mūla (transliterated to IAST, diacritics-folded) for the concept's stem and keep
    the verses that genuinely contain it, quoting the original line.
  * **Edges** are verse-grounded co-occurrences: an edge (A,B) exists iff some verse's
    mūla contains both stems; the edge cites those verses.
  * **Curated relations** are a separate, explicitly-typed layer encoding well-known
    doctrinal relations (e.g. sphoṭa manifested-by dhvani). Each MUST carry a verse
    anchor drawn from the empirical anchors — no free-floating scholarly assertions.

Nothing is anchored to a verse that does not contain the term; the quote makes every
claim checkable by hand.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from pranava.corpus.edition import build_edition
from pranava.corpus.translit import to_iast


def _fold(s: str) -> str:
    """Lowercase and strip combining marks so 'sphoṭa' matches 'sphoṭaḥ', 'sphoṭe', …."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
    ).lower()


@dataclass(frozen=True, slots=True)
class Concept:
    id: str
    iast: str
    stem: str  # folded stem used for matching
    gloss: str
    category: str  # ontology | phonology | semantics | grammar | metaphysics | epistemology


# The core technical vocabulary of the Vākyapadīya. Stems chosen to match inflected
# and compounded forms after diacritic-folding. Glosses are concise and standard.
CONCEPTS: list[Concept] = [
    Concept("sphota", "sphoṭa", "sphot", "the indivisible meaning-bearing linguistic whole that 'bursts' into cognition", "phonology"),
    Concept("dhvani", "dhvani", "dhvani", "the audible sound/utterance that manifests the sphoṭa", "phonology"),
    Concept("nada", "nāda", "nada", "resonance; the gross acoustic vibration", "phonology"),
    Concept("sabda", "śabda", "sabda", "word/language; for Bhartṛhari, the ultimate reality (śabda-tattva)", "metaphysics"),
    Concept("artha", "artha", "artha", "meaning/object; that which the word conveys", "semantics"),
    Concept("vak", "vāk", "vak", "speech; the faculty and levels of language", "metaphysics"),
    Concept("vakya", "vākya", "vakya", "sentence; the primary unit of meaning (akhaṇḍa-vākya)", "semantics"),
    Concept("pada", "pada", "pada", "word as a grammatical unit", "grammar"),
    Concept("varna", "varṇa", "varna", "phoneme/letter", "phonology"),
    Concept("jati", "jāti", "jati", "universal/class; the śabda-jāti view of meaning", "semantics"),
    Concept("vivarta", "vivarta", "vivarta", "apparent transformation of the one word-principle into the manifold", "metaphysics"),
    Concept("pratibha", "pratibhā", "pratibha", "the intuitive flash by which sentence-meaning is grasped", "epistemology"),
    Concept("kala", "kāla", "kala", "time; kāla-śakti orders action and manifestation", "metaphysics"),
    Concept("sakti", "śakti", "sakti", "power/capacity (e.g. of a word to signify)", "metaphysics"),
    Concept("brahma", "brahma", "brahma", "the absolute; identified with śabda-tattva", "metaphysics"),
    Concept("sambandha", "sambandha", "sambandha", "relation (esp. word–meaning relation)", "semantics"),
    Concept("kriya", "kriyā", "kriya", "action; verb-meaning, primary in the verb-first ontology", "semantics"),
    Concept("sadhana", "sādhana", "sadhana", "means/instrument; kāraka participants of an action", "grammar"),
    Concept("kala_buddhi", "buddhi", "buddhi", "cognition/intellect in which meaning appears", "epistemology"),
    Concept("upadhi", "upādhi", "upadhi", "limiting adjunct/condition", "semantics"),
]


@dataclass(frozen=True, slots=True)
class Anchor:
    vid: str
    quote: str  # original Devanāgarī mūla line containing the term


@dataclass(slots=True)
class ConceptNode:
    concept: Concept
    anchors: list[Anchor] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.concept.id


@dataclass(frozen=True, slots=True)
class Edge:
    src: str
    dst: str
    anchors: tuple[str, ...]  # verse ids where both concepts co-occur

    @property
    def weight(self) -> int:
        return len(self.anchors)


@dataclass(frozen=True, slots=True)
class CuratedRelation:
    src: str
    dst: str
    rel_type: str
    anchor_vid: str
    note: str = ""


# Doctrinal relations, each anchored to a verse chosen from the empirical anchors at
# build time (validated: the anchor must be a real co-occurrence or single-term anchor).
_CURATED = [
    ("sphota", "dhvani", "manifested-by", "the sphoṭa is revealed by the dhvani"),
    ("sabda", "brahma", "identified-with", "śabda-tattva is the absolute (śabdādvaita)"),
    ("sabda", "artha", "signifies", "the word–meaning relation"),
    ("vakya", "pratibha", "grasped-by", "sentence-meaning is grasped as pratibhā"),
    ("vak", "vivarta", "unfolds-by", "the one Word unfolds by vivarta into the manifold"),
    ("kriya", "sadhana", "requires", "an action is structured by its kāraka means"),
]


@dataclass(slots=True)
class ConceptGraph:
    nodes: dict[str, ConceptNode]
    edges: dict[tuple[str, str], Edge]
    _verse_terms: dict[str, set[str]]
    _curated: list[CuratedRelation]

    def get_edge(self, a: str, b: str) -> Edge | None:
        return self.edges.get(_ekey(a, b))

    def verse_terms(self, vid: str) -> set[str]:
        return self._verse_terms.get(vid, set())

    def anchors_for(self, concept_id: str) -> list[Anchor]:
        return self.nodes[concept_id].anchors

    def curated_relations(self) -> list[CuratedRelation]:
        return self._curated

    def to_dict(self) -> dict:
        return {
            "concepts": [
                {
                    "id": n.id,
                    "iast": n.concept.iast,
                    "gloss": n.concept.gloss,
                    "category": n.concept.category,
                    "anchor_count": len(n.anchors),
                    "anchors": [{"vid": a.vid, "quote": a.quote} for a in n.anchors[:8]],
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {"src": e.src, "dst": e.dst, "weight": e.weight, "anchors": list(e.anchors)}
                for e in self.edges.values()
            ],
            "curated_relations": [
                {"src": r.src, "dst": r.dst, "rel_type": r.rel_type,
                 "anchor_vid": r.anchor_vid, "note": r.note}
                for r in self._curated
            ],
        }


def _ekey(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


@lru_cache(maxsize=1)
def build_concept_graph() -> ConceptGraph:
    ed = build_edition()

    # Precompute folded IAST per verse and which concept stems each verse contains.
    verse_terms: dict[str, set[str]] = {}
    verse_lines: dict[str, list[str]] = {}
    for k in ed.karikas:
        folded_lines = [(line, _fold(to_iast(line))) for line in k.mula_lines]
        verse_lines[k.canonical] = [orig for orig, _ in folded_lines]
        present = {
            c.id
            for c in CONCEPTS
            if any(c.stem in folded for _, folded in folded_lines)
        }
        if present:
            verse_terms[k.canonical] = present

    # Nodes + anchors (quote the first mūla line in the verse that contains the stem).
    nodes: dict[str, ConceptNode] = {c.id: ConceptNode(c) for c in CONCEPTS}
    for k in ed.karikas:
        present = verse_terms.get(k.canonical)
        if not present:
            continue
        for c in CONCEPTS:
            if c.id not in present:
                continue
            quote = next(
                (line for line in k.mula_lines if c.stem in _fold(to_iast(line))),
                k.mula_lines[0],
            )
            nodes[c.id].anchors.append(Anchor(vid=k.canonical, quote=quote))

    # Co-occurrence edges.
    edges: dict[tuple[str, str], list[str]] = {}
    for vid, present in verse_terms.items():
        plist = sorted(present)
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                edges.setdefault(_ekey(plist[i], plist[j]), []).append(vid)
    edge_objs = {
        key: Edge(src=key[0], dst=key[1], anchors=tuple(sorted(vids)))
        for key, vids in edges.items()
    }

    # Curated relations — validate each has a real anchor (co-occurrence preferred,
    # else a verse anchoring at least the source concept). Drop any that cannot be anchored.
    curated: list[CuratedRelation] = []
    for src, dst, rtype, note in _CURATED:
        edge = edge_objs.get(_ekey(src, dst))
        if edge and edge.anchors:
            anchor_vid = edge.anchors[0]
        elif nodes[src].anchors:
            anchor_vid = nodes[src].anchors[0].vid
        else:
            continue
        curated.append(CuratedRelation(src, dst, rtype, anchor_vid, note))

    return ConceptGraph(nodes=nodes, edges=edge_objs, _verse_terms=verse_terms, _curated=curated)
