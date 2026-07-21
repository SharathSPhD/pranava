"""Precompute Parakeet feats for the Shrutilipi-sa corpus (resumable). Runs in a NeMo container."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
SH = ROOT / "data/alm/shrutilipi_sa"
FEATS = SH / "feats"


def main() -> int:
    from pranava.alm.data import read_wav
    from pranava.alm.encoder import ParakeetEncoder

    FEATS.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (SH / "manifest.jsonl").open(encoding="utf-8") if x.strip()]
    enc = ParakeetEncoder().load()
    done = skipped = 0
    bad = 0
    for r in rows:
        out = FEATS / f"{r['id']}.npy"
        if out.exists():
            skipped += 1
            continue
        wav, sr = read_wav(ROOT / r["wav"])
        if wav is None or len(wav) < int(0.1 * sr):  # zero/near-zero-length audio (data defect) — log & skip
            bad += 1
            print(json.dumps({"skip_empty_audio": r["id"], "n_samples": 0 if wav is None else len(wav),
                              "split": r.get("split")}), flush=True)
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
        if done % 500 == 0:
            print(json.dumps({"encoded": done, "skipped": skipped, "of": len(rows)}), flush=True)
    print(json.dumps({"encoded": done, "skipped": skipped, "total": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
