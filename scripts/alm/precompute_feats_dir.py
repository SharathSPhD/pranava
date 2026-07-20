"""Generic Parakeet feats precompute for any pranava corpus dir (manifest.jsonl + wav paths).

Usage (in a NeMo container): python precompute_feats_dir.py data/alm/librispeech [data/alm/vagdhenu …]
Resumable; skips defective audio with a log line (same guards as the shrutilipi variant).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]


def run(rel: str) -> None:
    import soundfile as sf  # reads FLAC (LibriSpeech) and WAV alike — pranava's read_wav is WAV-only
    from pranava.alm.encoder import ParakeetEncoder

    def read_any(p: Path):
        wav, sr = sf.read(str(p), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav.astype(np.float32), int(sr)

    base = ROOT / rel
    feats = base / "feats"
    feats.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (base / "manifest.jsonl").open(encoding="utf-8") if x.strip()]
    enc = ParakeetEncoder().load()
    done = skipped = bad = 0
    for r in rows:
        out = feats / f"{r['id']}.npy"
        if out.exists():
            skipped += 1
            continue
        try:
            wav, sr = read_any(ROOT / r["wav"])
        except Exception as e:
            bad += 1
            print(json.dumps({"skip_unreadable": r["id"], "err": str(e)[:80]}), flush=True)
            continue
        if wav is None or len(wav) < int(0.1 * sr):
            bad += 1
            print(json.dumps({"skip_empty_audio": r["id"]}), flush=True)
            continue
        f = enc.encode(wav, sr=sr)
        f = f.cpu().numpy() if hasattr(f, "cpu") else np.asarray(f)
        arr = np.squeeze(f).astype(np.float16)
        if arr.ndim != 2 or arr.shape[0] < 1:
            bad += 1
            print(json.dumps({"skip_bad_feat": r["id"], "shape": list(arr.shape)}), flush=True)
            continue
        np.save(out, arr)
        done += 1
        if done % 1000 == 0:
            print(json.dumps({"dir": rel, "encoded": done, "skipped": skipped, "of": len(rows)}), flush=True)
    print(json.dumps({"dir": rel, "encoded": done, "skipped": skipped, "bad": bad, "total": len(rows)}), flush=True)


if __name__ == "__main__":
    for rel in sys.argv[1:]:
        run(rel)
