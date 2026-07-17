"""Phase 3 — fit the Sphoṭa-Lens on the trained ALM and write its report.

Loads the trained projector (data/alm/projector.pt) + frozen core, builds fused audio+text samples
from val clips, fits the lens, and records the workspace band, the cross-modal fusion curve (peak =
where sphoṭa forms), and the articulation gradient (paśyantī→vaikharī). Run in prabhasa/nemo-gb10.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.data import CORPUS_DIR, load_manifest, text_to_bytes
from pranava.alm.projector import SphotaProjector
from pranava.sphota_lens.lens import fit_sphota_lens

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[1]
FEAT_DIR = CORPUS_DIR / "feats"
OUT = ROOT / "data" / "sphota_lens"


def main(n_samples: int = 40) -> int:
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev = core.torch_device
    bias = core.structural_bias

    # our own checkpoint; state_dict + ints only → weights_only is safe and correct
    blob = torch.load(ROOT / "data/alm/projector.pt", map_location=dev, weights_only=True)
    proj = SphotaProjector(d_enc=blob["d_enc"], d_model=blob["d_model"], downsample=4).to(dev).eval()
    proj.load_state_dict(blob["state_dict"])

    val = load_manifest("val")[:n_samples]
    samples = []
    with torch.no_grad():
        for ex in val:
            feats = np.load(FEAT_DIR / f"{ex.id}.npy").astype(np.float32)
            feats_t = torch.from_numpy(feats).unsqueeze(0).to(dev)
            audio_tok = proj(feats_t, structural_bias=bias)  # (1, Na, d)
            ids = torch.tensor([text_to_bytes(ex.text)], dtype=torch.long, device=dev)
            txt_emb = core.embed_tokens(ids)  # (1, Nt, d)
            emb = torch.cat([audio_tok, txt_emb], dim=1)
            samples.append((emb, int(audio_tok.shape[1])))

    report = fit_sphota_lens(core, samples)
    OUT.mkdir(parents=True, exist_ok=True)
    d = report.to_dict()

    # Steering demo: inject a concept direction into the workspace band during decode and show the
    # output shifts reproducibly (seeded, deterministic greedy). Direction points toward bytes 'aeiou'.
    from pranava.sphota_lens.steer import concept_direction, greedy_steered

    band = tuple(d["workspace_band"])
    prefix = samples[0][0][:, : samples[0][1], :]  # the audio tokens of val[0]
    direction = concept_direction(core, list(b"aeiou"))
    base = greedy_steered(core, prefix, band, None, 0.0, max_new=24)
    steered = [greedy_steered(core, prefix, band, direction, a, max_new=24) for a in (2.0, 4.0)]
    changed = [s != base for s in steered]
    # reproducibility: same alpha twice → identical output
    repro = greedy_steered(core, prefix, band, direction, 4.0, max_new=24) == steered[-1]
    d["steering"] = {
        "band": list(band),
        "output_changed_at_alpha": {"2.0": changed[0], "4.0": changed[1]},
        "any_shift": any(changed),
        "reproducible": bool(repro),
        "base_bytes": base[:16],
        "steered_bytes_alpha4": steered[-1][:16],
    }
    d["interpretation"] = {
        "workspace_band_layers": d["workspace_band"],
        "sphota_layer_fusion_peak": d["fusion_peak_layer"],
        "articulation_trend_rising": d["articulation_by_layer"][-1] > d["articulation_by_layer"][0],
        "note": "fusion peak = where audio (dhvani) and text (pada) representations converge into a "
                "unified meaning (sphoṭa); articulation rises paśyantī→vaikharī toward the head.",
    }
    (OUT / "sphota_lens_report.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(json.dumps({k: d[k] for k in
                      ("n_layers", "n_samples", "band_contrast", "workspace_band",
                       "fusion_peak_layer", "interpretation")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
