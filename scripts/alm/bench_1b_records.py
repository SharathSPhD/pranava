"""Per-clip leaderboard evidence for the 1.13B XL model — same standard as every other AA-gate row.

Loads the BEST xl1b checkpoint (saved by fair val CER), decodes the frozen 58-clip val under the fair
protocol (free greedy, 64-byte budget, stop at the model's own EOS), and writes per-clip predictions
into a records JSON to merge into data/benchmark/alm_vs_alm_records.json on the DGX (then host
_rescore_alm.py finalizes cer_norm/cer_raw identically for all models).

Run on the RTX 5090 (prabhasa/nemo-5090:26.02): python /work/pranava/scripts/alm/bench_1b_records.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pranava.alm.instruct import EOS, build_prefix
from pranava.alm.megatron_core import Megatron1BCore
from pranava.alm.megatron_lora import inject_megatron_lora, load_megatron_lora
from pranava.alm.projector import SphotaProjector

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
XL = ROOT / "data/alm/speech_corpus_indic_xl"
MODEL_NAME = "Śabda-ALM 1.13B+LoRA XL — free decode (ours)"


def main() -> int:
    blob = torch.load(ROOT / "data/alm/xl1b_ckpt.pt", map_location="cpu", weights_only=True)
    print(json.dumps({"ckpt_epoch": blob.get("epoch"), "ckpt_val_cer_norm": blob.get("val_cer_norm_fair")}))

    core = Megatron1BCore().load()
    for p in core._model.parameters():
        p.requires_grad_(False)
    inject_megatron_lora(core._model, r=int(blob.get("r", 16)))
    n = load_megatron_lora(core._model, blob["lora"])
    proj = SphotaProjector(d_enc=blob["d_enc"], d_model=blob["d_model"], downsample=4).to(core.torch_device)
    proj.load_state_dict(blob["projector"]); proj.eval()
    print(json.dumps({"lora_tensors_loaded": n}))

    rows = [json.loads(x) for x in (XL / "manifest.jsonl").open(encoding="utf-8") if x.strip()]
    val = [r for r in rows if r["split"] == "val"]
    bias = core.structural_bias
    per_clip = []
    with torch.no_grad():
        for r in val:
            feats = torch.from_numpy(np.load(XL / "feats" / f"{r['id']}.npy").astype(np.float32)
                                     ).unsqueeze(0).to(core.torch_device)
            audio = proj(feats, structural_bias=bias)
            prefix = build_prefix(core, audio, "transcribe the speech")
            out = core.greedy_from_embeds(prefix, max_new=64, stop_token=EOS)
            pred = bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()
            per_clip.append({"id": r["id"], "pred": pred, "gold": r["text"]})
            print(f"  {r['id']}: {pred[:56]!r}", flush=True)

    out_p = ROOT / "data/alm/xl1b_val_records.json"
    out_p.write_text(json.dumps({MODEL_NAME: per_clip}, indent=2, ensure_ascii=False))
    print(f"wrote {out_p} ({len(per_clip)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
