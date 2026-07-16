"""Build a validated, aligned edition of the Vākyapadīya.

Sources (local, scraped/authored by the operator):
  - mūla (Devanāgarī):  docs/.../vakyapadiya_mula_deva.json
  - translation+commentary: docs/.../build/commentary/kanda{1,2,3}.json

The builder aligns commentary onto mūla verses by canonical verse id. It is
deliberately conservative: verses without an aligned commentary entry keep
``translation = None`` — no placeholder is ever fabricated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pranava.corpus.verse_id import VerseId, parse_verse_label

_DOCS = Path(__file__).resolve().parents[3] / "docs" / "vakyapadiyam-various-sources"
_MULA = _DOCS / "vakyapadiya_mula_deva.json"
_COMMENTARY = _DOCS / "build" / "commentary"


@dataclass(frozen=True, slots=True)
class Karika:
    vid: VerseId
    mula_lines: tuple[str, ...]
    href: str | None = None
    section: str | None = None
    translation: str | None = None
    commentary: str | None = None
    contested: bool = False

    @property
    def canonical(self) -> str:
        return self.vid.canonical


@dataclass(slots=True)
class Edition:
    karikas: list[Karika] = field(default_factory=list)
    _index: dict[str, Karika] = field(default_factory=dict)

    def get(self, canonical: str) -> Karika | None:
        return self._index.get(canonical)

    def coverage(self) -> dict[str, float | int]:
        n = len(self.karikas)
        translated = sum(1 for k in self.karikas if k.translation is not None)
        commented = sum(1 for k in self.karikas if k.commentary is not None)
        contested = sum(1 for k in self.karikas if k.contested)
        return {
            "karikas": n,
            "translated": translated,
            "commented": commented,
            "contested": contested,
            "translated_fraction": (translated / n) if n else 0.0,
        }


def _load_commentary() -> dict[str, dict]:
    """Return {canonical_verse_id: entry} merged across the three kāṇḍa files.

    Commentary keys look like "Verse 1.1", "Verse 2.1-2", "Verse 3.1.1".
    """
    merged: dict[str, dict] = {}
    for k in (1, 2, 3):
        path = _COMMENTARY / f"kanda{k}.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        for key, val in raw.items():
            if key == "_note" or not isinstance(val, dict):
                continue
            try:
                vid = parse_verse_label(key)
            except ValueError:
                continue
            merged[vid.canonical] = val
    return merged


@lru_cache(maxsize=1)
def build_edition() -> Edition:
    mula = json.loads(_MULA.read_text(encoding="utf-8"))
    commentary = _load_commentary()

    edition = Edition()
    for entry in mula["verses"]:
        lines = entry.get("lines") or []
        if not lines:  # section header, not a verse
            continue
        try:
            vid = parse_verse_label(entry["label"])
        except ValueError:
            continue
        comm = commentary.get(vid.canonical)
        translation = commentary_field(comm, "translation")
        karika = Karika(
            vid=vid,
            mula_lines=tuple(lines),
            href=entry.get("href"),
            section=commentary_field(comm, "section"),
            translation=translation,
            commentary=commentary_field(comm, "commentary"),
            contested=bool(comm.get("contested")) if comm else False,
        )
        edition.karikas.append(karika)
        edition._index[vid.canonical] = karika

    edition.karikas.sort(key=lambda k: k.vid)
    return edition


def commentary_field(comm: dict | None, name: str) -> str | None:
    if not comm:
        return None
    val = comm.get(name)
    if val is None:
        return None
    val = str(val).strip()
    return val or None
