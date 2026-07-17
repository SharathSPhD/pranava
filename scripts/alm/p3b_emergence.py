"""Sphoṭa-Lens v2 — fit the meaning-emergence lens on the trained ALM and validate the sphoṭa layer.

Locates, correlationally AND causally, the layer where the sentence's meaning (kriyā) becomes
decodable from the audio-token positions — where sound bursts into meaning. Run in prabhasa/nemo-gb10.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.data import CORPUS_DIR, load_manifest, text_to_bytes
from pranava.alm.projector import SphotaProjector
from pranava.sphota_lens.emergence import meaning_emergence

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[1]
FEAT_DIR = CORPUS_DIR / "feats"
OUT = ROOT / "data" / "sphota_lens"


def main(n: int = 240, min_per_class: int = 6) -> int:
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias
    blob = torch.load(ROOT / "data/alm/projector.pt", map_location=dev, weights_only=True)
    proj = SphotaProjector(d_enc=blob["d_enc"], d_model=blob["d_model"], downsample=4).to(dev).eval()
    proj.load_state_dict(blob["state_dict"])

    exs = [e for e in load_manifest() if e.kriya and (FEAT_DIR / f"{e.id}.npy").exists()]
    # keep classes with enough support for a grouped-CV probe
    from collections import Counter
    ct = Counter(e.kriya for e in exs)
    exs = [e for e in exs if ct[e.kriya] >= min_per_class][:n]

    samples = []
    with torch.no_grad():
        for ex in exs:
            feats = torch.from_numpy(np.load(FEAT_DIR / f"{ex.id}.npy").astype(np.float32)).unsqueeze(0).to(dev)
            audio_tok = proj(feats, structural_bias=bias)
            ids = torch.tensor([text_to_bytes(ex.text)], dtype=torch.long, device=dev)
            emb = torch.cat([audio_tok, core.embed_tokens(ids)], dim=1)
            samples.append((emb, int(audio_tok.shape[1]), ex.kriya, ex.template))

    report = meaning_emergence(core, samples)
    OUT.mkdir(parents=True, exist_ok=True)
    d = report.to_dict()
    d["reading"] = {
        "sphota_layer": d["sphota_layer"],
        "peak_decodability": round(max(x for x in d["decodability_by_layer"] if not np.isnan(x)), 4),
        "chance": d["chance"],
        "validated": d["validated"],
        "meaning": "layer where the sentence's kriyā becomes decodable from AUDIO positions "
                   "(correlational peak) AND whose ablation most hurts meaning (causal peak); "
                   "agreement within ±1 layer validates the sphoṭa locus.",
    }
    (OUT / "emergence_report.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(json.dumps({k: d[k] for k in ("n_samples", "n_classes", "chance", "peak_layer",
                                        "causal_peak_layer", "sphota_layer", "validated")}, indent=2))
    print("decodability_by_layer:", [round(x, 3) for x in d["decodability_by_layer"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
