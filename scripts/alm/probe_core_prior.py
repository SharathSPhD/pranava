"""Why does Śabda-ALM transcribe Sanskrit but babble English? Probe the FROZEN core's language prior.

Three bilingual training levers (more epochs, 4x LoRA rank, 2x English weighting) all left English
val CER at ~0.80 while Sanskrit sat at ~0.51. Two rival explanations:

  H-capacity  the adapter is too small to learn audio→English conditioning   (falsified by v4/v5)
  H-prior     the frozen 1.13B core is Sanskrit-only, so English asks LoRA to install an English
              language model AND audio conditioning through a rank-r bottleneck

This probe decides it WITHOUT audio: teacher-forced byte cross-entropy of the core on gold text
alone (the generous case — no acoustic uncertainty at all). If English CE is far above Sanskrit CE
even with the trained LoRA loaded, the ceiling is the core's monolingual prior, not adapter capacity,
and English parity is infeasible under a frozen Sanskrit core — a structural finding, not a tuning bug.

Run on the RTX 5090:  python /work/pranava/scripts/alm/probe_core_prior.py
Artifact: data/alm/core_prior_probe.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from pranava.alm.instruct import EOS
from pranava.alm.megatron_core import Megatron1BCore
from pranava.alm.megatron_lora import inject_megatron_lora, load_megatron_lora

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
SETS = {"english": ROOT / "data/alm/librispeech", "sanskrit": ROOT / "data/alm/shrutilipi_sa"}
N = 200


def texts(name: str, split: str = "validation") -> list[str]:
    p = SETS[name] / "manifest.jsonl"
    rows = [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]
    out = [r["text"] for r in rows if r["split"] == split and r["text"].strip()]
    if len(out) < N:  # validation split may be small — fall back to train text
        out += [r["text"] for r in rows if r["split"] == "train" and r["text"].strip()]
    return random.Random(0).sample(out, min(N, len(out)))


@torch.no_grad()
def mean_ce(core, sents: list[str], dev) -> dict:
    """Teacher-forced next-byte cross-entropy (nats/byte) over gold text — no audio, no prefix."""
    ce = torch.nn.CrossEntropyLoss()
    vals = []
    for s in sents:
        ids = torch.tensor([list(s.encode("utf-8"))[:420] + [EOS]], dtype=torch.long, device=dev)
        if ids.shape[1] < 2:
            continue
        logits = core.logits_from_embeds(core.embed_tokens(ids[:, :-1]))
        vals.append(float(ce(logits[0], ids[0, 1:])))
    t = torch.tensor(vals)
    return {"nats_per_byte": round(float(t.mean()), 4), "std": round(float(t.std()), 4),
            "bits_per_byte": round(float(t.mean()) / 0.6931, 4), "n": len(vals)}


def main() -> int:
    core = Megatron1BCore().load()
    for p in core._model.parameters():
        p.requires_grad_(False)
    dev = core.torch_device
    en, sa = texts("english"), texts("sanskrit")

    res = {"probe": "teacher-forced byte cross-entropy of the frozen core on GOLD TEXT (no audio)",
           "why": "separates H-prior (core is Sanskrit-only) from H-capacity (adapter too small)",
           "n_per_language": N, "conditions": {}}

    res["conditions"]["frozen_core_no_lora"] = {"english": mean_ce(core, en, dev),
                                                "sanskrit": mean_ce(core, sa, dev)}
    print(json.dumps(res["conditions"]["frozen_core_no_lora"], indent=2), flush=True)

    ck = ROOT / "data/alm/bi1b_ckpt.pt"
    if ck.exists():
        blob = torch.load(ck, map_location=dev, weights_only=True)
        r = int(blob.get("r", 16))
        inject_megatron_lora(core._model, r=r)
        n = load_megatron_lora(core._model, blob["lora"])
        res["lora"] = {"ckpt": ck.name, "r": r, "tensors": n,
                       "val_cer_norm_fair": blob.get("val_cer_norm_fair")}
        res["conditions"]["with_trained_lora"] = {"english": mean_ce(core, en, dev),
                                                  "sanskrit": mean_ce(core, sa, dev)}
        print(json.dumps(res["conditions"]["with_trained_lora"], indent=2), flush=True)

    base = res["conditions"]["frozen_core_no_lora"]
    res["verdict"] = {
        "en_over_sa_nats_frozen": round(base["english"]["nats_per_byte"] - base["sanskrit"]["nats_per_byte"], 4),
        "reading": "A large positive gap means the frozen core assigns English text far lower "
                   "probability than Sanskrit text — LoRA must supply the English language model "
                   "itself, on top of audio conditioning. That is H-prior: an architectural ceiling "
                   "of a frozen monolingual core, not an adapter-capacity shortfall.",
    }
    (ROOT / "data/alm/core_prior_probe.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
