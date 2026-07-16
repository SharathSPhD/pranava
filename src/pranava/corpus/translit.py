"""Devanāgarī↔IAST transliteration and text-integrity helpers.

Wraps ``indic_transliteration.sanscript`` (a standard, well-tested scheme) and
adds edition-specific hygiene: stripping verse-number decoration and measuring
how much of a line actually lies in the Devanāgarī Unicode block (mojibake
detector).
"""
from __future__ import annotations

import re
import unicodedata

from indic_transliteration import sanscript

# Devanāgarī digits ० .. ९ and the double daṇḍa that close a kārikā: "॥ ११३ ॥"
_VERSE_MARKER_RE = re.compile(r"॥[^॥]*॥\s*$")
_DANDA_RE = re.compile(r"[।॥]")
_DEVA_DIGITS = re.compile(r"[०-९]")


def strip_verse_markers(line: str) -> str:
    """Remove the trailing ``॥ <number> ॥`` decoration and stray daṇḍas/numbers."""
    line = _VERSE_MARKER_RE.sub("", line)
    line = _DANDA_RE.sub("", line)
    line = _DEVA_DIGITS.sub("", line)
    return line.strip()


def to_iast(line: str) -> str:
    """Transliterate a Devanāgarī line to IAST.

    Verse markers are stripped first so numbers don't turn into Latin digits;
    the daṇḍa is dropped from the output rather than rendered as ``|``.
    """
    cleaned = strip_verse_markers(line)
    out = sanscript.transliterate(cleaned, sanscript.DEVANAGARI, sanscript.IAST)
    return out.replace("|", "").strip()


def devanagari_ratio(text: str) -> float:
    """Fraction of *letter* characters that are in the Devanāgarī block.

    Whitespace, punctuation, and digits are ignored so that ``॥ १ ॥`` decoration
    doesn't distort the score. Returns 0.0 for text with no letters.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    deva = sum(1 for c in letters if "DEVANAGARI" in unicodedata.name(c, ""))
    return deva / len(letters)
