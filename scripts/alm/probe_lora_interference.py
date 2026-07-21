"""Does the shared LoRA INTERFERE with English? Sweep adapter scale per language.

core_prior_probe.json falsified both earlier hypotheses and found something better: on identical
text-only conditions the trained bilingual adapter IMPROVES Sanskrit byte-modelling (2.675 -> 1.507
nats/byte) but DEGRADES English (1.873 -> 3.841) — worse than the untouched core. The English ceiling
is therefore not missing capacity (v4/v5 falsified that) and not a missing English prior (the frozen
core is *better* at English than at Sanskrit); it is language interference inside one shared adapter.

Prediction, if that diagnosis is right: attenuating the adapter should IMPROVE English transcription
and HURT Sanskrit — a crossing pair of curves. This sweeps LoRA scale s in {1, .75, .5, .25, 0} and
evaluates real audio transcription on both validation sets under the fair protocol.

A crossing means language-conditional adapter routing (per-language scale, selected by the instruction
that already accompanies every turn) is a real, cheap lever — not a retrain.

Run on the RTX 5090:  python /work/pranava/scripts/alm/probe_lora_interference.py
Artifact: data/alm/lora_interference_probe.json
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from pranava.alm.megatron_core import Megatron1BCore
from pranava.alm.megatron_lora import inject_megatron_lora, load_megatron_lora
from pranava.alm.projector import SphotaProjector

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "train_bi", Path(__file__).resolve().parent / "train_1b_bi.py")
_tb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tb)

ROOT = _tb.ROOT
SCALES = [1.0, 0.75, 0.5, 0.25, 0.0]
LIMIT = 120  # clips per language per scale — enough to separate 0.80 from a material change


def set_lora_scale(model, mult: float, base: dict) -> None:
    for name, mod in model.named_modules():
        pair = getattr(mod, "lora_pair", None)
        if pair is not None:
            pair.scale = base[name] * mult


def main() -> int:
    core = Megatron1BCore().load()
    for p in core._model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias

    ck = ROOT / "data/alm/bi1b_ckpt.pt"
    blob = torch.load(ck, map_location=dev, weights_only=True)
    r = int(blob.get("r", 16))
    inject_megatron_lora(core._model, r=r)
    n = load_megatron_lora(core._model, blob["lora"])
    proj = SphotaProjector(d_enc=int(blob["d_enc"]), d_model=core.d_model, downsample=4).to(dev)
    proj.load_state_dict(blob["projector"])
    proj.eval()

    base = {name: mod.lora_pair.scale for name, mod in core._model.named_modules()
            if getattr(mod, "lora_pair", None) is not None}

    val_en = _tb.load_split("librispeech", "validation")
    val_sa = _tb.load_split("shrutilipi", "validation")

    res = {"probe": "LoRA-scale sweep on real audio (fair protocol) — tests the interference diagnosis",
           "ckpt": {"name": ck.name, "r": r, "tensors": n,
                    "val_cer_norm_fair": blob.get("val_cer_norm_fair")},
           "limit_per_language": LIMIT, "scales": {}}

    for s in SCALES:
        set_lora_scale(core._model, s, base)
        en = _tb.eval_fair(core, proj, bias, val_en, dev, slp1=False, limit=LIMIT)
        sa = _tb.eval_fair(core, proj, bias, val_sa, dev, slp1=True, limit=LIMIT)
        macro = round((en["cer"] + sa["cer"]) / 2, 4) if en["cer"] is not None and sa["cer"] is not None else None
        res["scales"][str(s)] = {"en_val": en, "sa_val": sa, "macro_cer": macro}
        print(json.dumps({"lora_scale": s} | res["scales"][str(s)]), flush=True)

    en_by = {s: res["scales"][str(s)]["en_val"]["cer"] for s in SCALES}
    sa_by = {s: res["scales"][str(s)]["sa_val"]["cer"] for s in SCALES}
    best_en = min(en_by, key=lambda k: en_by[k])
    best_sa = min(sa_by, key=lambda k: sa_by[k])
    res["verdict"] = {
        "best_scale_english": best_en, "best_en_cer": en_by[best_en],
        "best_scale_sanskrit": best_sa, "best_sa_cer": sa_by[best_sa],
        "interference_confirmed": bool(best_en != best_sa and best_en < 1.0),
        "reading": "If English prefers an attenuated adapter while Sanskrit prefers the full one, "
                   "the two languages are fighting over one adapter and per-language routing is the "
                   "fix. If English is flat or best at scale 1.0, interference is NOT the mechanism "
                   "and the English ceiling stands as an honest architectural limit.",
    }
    (ROOT / "data/alm/lora_interference_probe.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res["verdict"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
