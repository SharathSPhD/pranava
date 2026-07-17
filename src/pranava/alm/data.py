"""Paired speech↔text dataset for ALM training (Phase 1 output).

Reads the manifest produced by scripts/alm/build_speech_corpus.py. Loading the manifest and the
WAV files needs no GPU/NeMo, so this is host-testable; feature extraction happens at train time.
"""
from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


import os

# Which corpus to use — swap to the native-Sanskrit (indic-parler-tts) corpus via
# PRANAVA_CORPUS=speech_corpus_indic. Default is the original controlled TTS corpus.
CORPUS_NAME = os.environ.get("PRANAVA_CORPUS", "speech_corpus")


def _root() -> Path:
    for c in ("/work/pranava", "/home/sharaths/projects/pranava"):
        if (Path(c) / "data/alm" / CORPUS_NAME / "manifest.jsonl").exists():
            return Path(c)
    return Path(__file__).resolve().parents[3]


CORPUS_DIR = _root() / "data" / "alm" / CORPUS_NAME
MANIFEST = CORPUS_DIR / "manifest.jsonl"


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    text: str
    wav_path: Path
    karaka: list
    split: str
    duration_s: float

    @property
    def kriya(self) -> str | None:
        """The verb (kriyā) — the sentence's core meaning, from the gold kāraka parse."""
        for pair in self.karaka:
            if len(pair) == 2 and pair[1] == "kriyA":
                return pair[0]
        return None

    @property
    def template(self) -> str:
        """A leakage-safe CV group: the sentence with its kāraka fillers blanked."""
        fillers = {p[0] for p in self.karaka if len(p) == 2}
        return " ".join("_" if w in fillers else w for w in self.text.split())


def load_manifest(split: str | None = None) -> list[Example]:
    rows = [json.loads(x) for x in MANIFEST.open(encoding="utf-8") if x.strip()]
    out = []
    for r in rows:
        if split and r["split"] != split:
            continue
        out.append(
            Example(
                id=r["id"], text=r["text"], wav_path=_root() / r["wav"],
                karaka=r.get("karaka", []), split=r["split"],
                duration_s=float(r.get("duration_s", 0.0)),
            )
        )
    return out


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr, raw = wf.getframerate(), wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def text_to_bytes(text: str) -> list[int]:
    """Byte ids for the core's byte-level tokenizer (matches vocab_size 256)."""
    return list(text.encode("utf-8"))
