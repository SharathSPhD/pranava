"""Prepare LibriSpeech (public, openslr.org/12) → pranava corpus format — the ENGLISH arena.

Operator hypothesis: Śabda-ALM must stand head-to-head on the baselines' home language. LibriSpeech
test-clean is THE standard public English ASR benchmark; train-clean-100 (100 h) is the training set.
Splits are the official ones — test-clean is never touched in training.

Targets: lowercase ASCII English (byte-core native; the scoring fold lowercases everything anyway,
and LibriSpeech references are caseless by convention). FLAC decoded to 16 kHz wav on the fly by the
feats step — the manifest points at the flac files (soundfile reads flac natively).

Writes data/alm/librispeech/{manifest.jsonl, datasheet.json}.
Run: .venv/bin/python scripts/alm/prep_librispeech.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LS = ROOT / "data/librispeech/LibriSpeech"
OUT = ROOT / "data/alm/librispeech"

SPLITS = {"train-clean-100": "train", "dev-clean": "validation", "test-clean": "test"}


def main() -> int:
    import soundfile as sf

    OUT.mkdir(parents=True, exist_ok=True)
    rows, n = [], {"train": 0, "validation": 0, "test": 0}
    hours = {"train": 0.0, "validation": 0.0, "test": 0.0}
    for ls_split, split in SPLITS.items():
        base = LS / ls_split
        if not base.exists():
            raise SystemExit(f"{base} missing — download not finished?")
        for trans in sorted(base.rglob("*.trans.txt")):
            for line in trans.open(encoding="utf-8"):
                utt_id, _, text = line.strip().partition(" ")
                flac = trans.parent / f"{utt_id}.flac"
                if not flac.exists() or not text:
                    continue
                info = sf.info(str(flac))
                dur = info.frames / info.samplerate
                cid = f"ls_{split[:2]}_{n[split]:06d}"
                rows.append({"id": cid, "text": text.strip().lower(),
                             "wav": str(flac.relative_to(ROOT)), "sr": int(info.samplerate),
                             "duration_s": round(dur, 3), "split": split, "lang": "en",
                             "source": f"LibriSpeech/{ls_split}"})
                n[split] += 1
                hours[split] += dur / 3600
    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    ds = {"counts": n, "hours": {k: round(v, 2) for k, v in hours.items()},
          "provenance": "LibriSpeech (openslr.org/12, CC-BY-4.0): train-clean-100 → train; dev-clean → validation; "
                        "test-clean → test (official splits, test untouched in training)",
          "text": "lowercase ASCII (byte-core native; scoring fold lowercases all systems identically)"}
    (OUT / "datasheet.json").write_text(json.dumps(ds, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ds, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
