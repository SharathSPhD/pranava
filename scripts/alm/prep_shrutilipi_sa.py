"""Prepare Shrutilipi-Sanskrit (amithm3/shrutilipi, config sa) → pranava corpus format.

REAL human Sanskrit speech (AI4Bharat Shrutilipi: All-India-Radio broadcasts; Apache-2.0), with the
public train/validation/test splits preserved EXACTLY (test stays untouched until the final eval).
TRIZ-informed prep: (P2 Taking out) quality-filter the train split only — duration/length-ratio
outliers and non-Devanagari junk hurt batch-1 training; test/val are NEVER filtered. (P32 Change
representation) transcripts are transliterated Devanagari→SLP1 (the byte-core's native scheme);
the original Devanagari is kept alongside for scoring/display.

Reads the parquet files snapshot-downloaded to data/shrutilipi_sa/, writes:
  data/alm/shrutilipi_sa/{wav/*.wav, manifest.jsonl, datasheet.json}

Run on host: .venv/bin/python scripts/alm/prep_shrutilipi_sa.py
"""
from __future__ import annotations

import io
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/shrutilipi_sa"
OUT = ROOT / "data/alm/shrutilipi_sa"
WAV = OUT / "wav"

_DEVA = re.compile(r"[ऀ-ॿ]")
_KEEP = re.compile(r"[^ऀ-ॿ\s]")  # anything not Devanagari/space (numerals, latin, punct)


def _clean_deva(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = _KEEP.sub(" ", t)  # drop punctuation/latin/digits — same rule later applied to ALL model outputs
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main() -> int:
    import soundfile as sf
    from indic_transliteration import sanscript

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("pip install pyarrow")

    files = sorted(SRC.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {SRC} — download not finished?")
    WAV.mkdir(parents=True, exist_ok=True)

    rows_out, stats = [], {"train": 0, "validation": 0, "test": 0, "filtered_train": 0}
    for f in files:
        # split name from path (…/sa/train-00001-of-00013.parquet or …/train/…)
        name = f.name.lower()
        split = ("test" if "test" in name or "/test/" in str(f).lower()
                 else "validation" if "valid" in name or "/validation/" in str(f).lower()
                 else "train")
        tbl = pq.read_table(f)
        cols = {c.lower(): c for c in tbl.column_names}
        audio_col = cols.get("audio")
        text_col = cols.get("transcription") or cols.get("transcriptions") or cols.get("text")
        d = tbl.to_pylist()
        for i, r in enumerate(d):
            raw = r[text_col]
            deva = _clean_deva(raw if isinstance(raw, str) else str(raw))
            if not deva or not _DEVA.search(deva):
                if split == "train":
                    stats["filtered_train"] += 1
                    continue
                deva = deva or "-"
            a = r[audio_col]
            # HF parquet audio: {'bytes': flac/wav bytes, 'path': …} or {'array','sampling_rate'}
            if isinstance(a, dict) and a.get("bytes"):
                wav_arr, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
            elif isinstance(a, dict) and a.get("array") is not None:
                wav_arr, sr = np.asarray(a["array"], dtype=np.float32), int(a["sampling_rate"])
            else:
                if split == "train":
                    stats["filtered_train"] += 1
                    continue
                raise SystemExit(f"unreadable audio in {f} row {i} (split={split})")
            if wav_arr.ndim > 1:
                wav_arr = wav_arr.mean(axis=1)
            dur = len(wav_arr) / sr
            slp1 = sanscript.transliterate(deva, sanscript.DEVANAGARI, sanscript.SLP1)
            # TRIZ P2: TRAIN-only quality extraction (never touch val/test)
            if split == "train":
                if not (0.5 <= dur <= 25.0) or not (3 <= len(slp1) <= 400):
                    stats["filtered_train"] += 1
                    continue
            cid = f"sh_{split[:2]}_{stats[split]:05d}"
            wp = WAV / f"{cid}.wav"
            sf.write(str(wp), wav_arr, sr)
            rows_out.append({"id": cid, "text": slp1, "text_devanagari": deva,
                             "wav": str(wp.relative_to(ROOT)), "sr": int(sr),
                             "duration_s": round(dur, 3), "split": split,
                             "source": "amithm3/shrutilipi:sa"})
            stats[split] += 1
        print(json.dumps({"file": f.name, "split": split, "total_so_far": stats}), flush=True)

    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows_out) + "\n", encoding="utf-8")
    hours = {s: round(sum(r["duration_s"] for r in rows_out if r["split"] == s) / 3600, 2)
             for s in ("train", "validation", "test")}
    ds = {"counts": stats, "hours": hours,
          "provenance": "amithm3/shrutilipi (HF mirror of AI4Bharat Shrutilipi), config sa — REAL human "
                        "Sanskrit (All India Radio). Public splits preserved; TEST NEVER FILTERED/TOUCHED.",
          "text": "Devanagari cleaned (punct/digits/latin→space, NFC) then SLP1-transliterated (core-native). "
                  "The same cleaning is applied to every model's output at scoring time.",
          "train_filter": "duration 0.5–25s and 3–400 SLP1 chars (train only)"}
    (OUT / "datasheet.json").write_text(json.dumps(ds, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ds, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
