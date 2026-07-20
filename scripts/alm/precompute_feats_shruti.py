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
    for r in rows:
        out = FEATS / f"{r['id']}.npy"
        if out.exists():
            skipped += 1
            continue
        wav, sr = read_wav(ROOT / r["wav"])
        f = enc.encode(wav, sr=sr)
        f = f.cpu().numpy() if hasattr(f, "cpu") else np.asarray(f)
        np.save(out, np.squeeze(f).astype(np.float16))
        done += 1
        if done % 500 == 0:
            print(json.dumps({"encoded": done, "skipped": skipped, "of": len(rows)}), flush=True)
    print(json.dumps({"encoded": done, "skipped": skipped, "total": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
