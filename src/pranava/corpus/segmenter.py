"""Sandhi/compound segmentation via vidyut-cheda (Ambuda's Rust engine).

The Saṃsādhanī container's own segmenter is missing its Heritage backend (see
research/01-samsaadhanii-integration.md), so we use vidyut-cheda for splitting and keep
Saṃsādhanī's authoritative morph.cgi for *validation*. vidyut is statistical and makes
mistakes; validating each proposed pāda against Saṃsādhanī turns those mistakes into
honest coverage misses rather than silent errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).resolve().parents[3] / ".vidyut-data"


def data_available() -> bool:
    return (_DATA / "cheda" / "model.msgpack").exists()


@dataclass(frozen=True, slots=True)
class Pada:
    text_dev: str  # segmented word, Devanāgarī
    text_slp: str  # SLP1
    lemma: str  # vidyut's proposed lemma (may be empty/incorrect)


@lru_cache(maxsize=1)
def _chedaka():
    from vidyut.cheda import Chedaka

    return Chedaka(str(_DATA))


def _to_slp(dev: str) -> str:
    from vidyut.lipi import Scheme, transliterate

    return transliterate(dev, Scheme.Devanagari, Scheme.Slp1)


def _to_dev(slp: str) -> str:
    from vidyut.lipi import Scheme, transliterate

    return transliterate(slp, Scheme.Slp1, Scheme.Devanagari)


def segment_line(line_dev: str) -> list[Pada]:
    """Segment a Devanāgarī line (already stripped of verse markers) into pādas."""
    from pranava.corpus.translit import strip_verse_markers

    slp = _to_slp(strip_verse_markers(line_dev))
    if not slp.strip():
        return []
    out: list[Pada] = []
    for tok in _chedaka().run(slp):
        lemma = ""
        info = getattr(tok, "info", None)
        if info is not None:
            lemma = str(getattr(info, "lemma", "") or "")
        out.append(Pada(text_dev=_to_dev(tok.text), text_slp=tok.text, lemma=lemma))
    return out
