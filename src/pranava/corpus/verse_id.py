"""Canonical, sortable identifiers for Vākyapadīya kārikās.

A :class:`VerseId` normalises the several label shapes found in the source
edition into one comparable, hashable value while preserving every
distinction (kāṇḍa, samuddeśa section, verse range, half-verse pāda).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import total_ordering

# "Verse 3.7.113" / "Verse 1.5" / "Verse 1.16-17" / "Verse 2.127cd"
_LABEL_RE = re.compile(
    r"^Verse\s+"
    r"(?P<kanda>[123])"
    r"\.(?P<a>\d+)"
    r"(?:\.(?P<b>\d+))?"
    r"(?:-(?P<end>\d+))?"
    r"(?:\((?P<phalf>[a-d]{1,2})\)|(?P<half>[a-d]{1,2}))?"
    r"\s*$"
)


@total_ordering
@dataclass(frozen=True, slots=True)
class VerseId:
    kanda: int
    section: int | None
    start: int
    end: int
    half: str | None = None

    @property
    def is_range(self) -> bool:
        return self.end != self.start

    @property
    def canonical(self) -> str:
        if self.section is not None:
            head = f"{self.kanda}.{self.section}.{self.start}"
        else:
            head = f"{self.kanda}.{self.start}"
        if self.is_range:
            head += f"-{self.end}"
        if self.half:
            head += self.half
        return head

    # ordering key: kāṇḍa, section (0 if flat), start, end, half
    def _key(self) -> tuple[int, int, int, int, str]:
        return (self.kanda, self.section or 0, self.start, self.end, self.half or "")

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, VerseId):
            return NotImplemented
        return self._key() < other._key()

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.canonical


def parse_verse_label(label: str) -> VerseId:
    """Parse a source label like ``"Verse 3.7.113"`` into a :class:`VerseId`.

    Raises :class:`ValueError` for anything that is not a verse label
    (e.g. book/section headers).
    """
    m = _LABEL_RE.match(label.strip())
    if not m:
        raise ValueError(f"not a verse label: {label!r}")
    kanda = int(m["kanda"])
    a = int(m["a"])
    b = m["b"]
    if b is not None:
        section: int | None = a
        start = int(b)
    else:
        section = None
        start = a
    end = int(m["end"]) if m["end"] else start
    half = m["half"] or m["phalf"]
    return VerseId(kanda=kanda, section=section, start=start, end=end, half=half)
