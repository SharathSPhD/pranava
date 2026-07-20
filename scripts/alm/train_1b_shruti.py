"""Adapt the 1.13B Śabda-ALM to REAL human Sanskrit speech (Shrutilipi-sa) — the public-benchmark run.

TRIZ P6 (universality/curriculum): warm-starts BOTH the projector and the LoRA from xl1b_ckpt.pt — the
model that already aligns audio→core on synthetic Sanskrit — then adapts to broadcast speech.
Transcribe-only (Shrutilipi has no kāraka labels). Targets are SLP1 (P32: the core's native scheme).
Fair eval on the PUBLIC validation split each epoch (free greedy, stop at EOS, max_new 384 — real
sentences are long); the PUBLIC TEST SPLIT IS NEVER TOUCHED here (final eval only, eval_shrutilipi.py).

Artifacts: data/alm/sh1b_ckpt.pt (best-by-val), data/alm/sh1b_metrics.json.
Run on the RTX 5090 (prabhasa/nemo-5090:26.02, cwd /work):
  python /work/pranava/scripts/alm/train_1b_shruti.py [epochs]
"""
from __future__ import annotations

import json
import random
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

from pranava.alm.instruct import EOS, build_prefix
from pranava.alm.megatron_core import Megatron1BCore
from pranava.alm.megatron_lora import (inject_megatron_lora, load_megatron_lora,
                                       megatron_lora_state_dict)
from pranava.alm.projector import SphotaProjector

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
SH = ROOT / "data/alm/shrutilipi_sa"
FEATS = SH / "feats"

_SLP1_IAST = {"A": "ā", "I": "ī", "U": "ū", "f": "ṛ", "F": "ṝ", "x": "ḷ", "X": "ḹ", "E": "ai",
              "O": "au", "M": "ṃ", "H": "ḥ", "K": "kh", "G": "gh", "N": "ṅ", "C": "ch", "J": "jh",
              "Y": "ñ", "w": "ṭ", "W": "ṭh", "q": "ḍ", "Q": "ḍh", "R": "ṇ", "T": "th", "D": "dh",
              "P": "ph", "B": "bh", "S": "ś", "z": "ṣ", "L": "ḷ"}


def _norm(s: str) -> str:
    s = "".join(_SLP1_IAST.get(c, c) for c in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = "".join(c if (("a" <= c <= "z") or ("0" <= c <= "9") or c.isspace()) else " " for c in s)
    return " ".join(s.split())


def _lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        cur = [i + 1]
        for j, c2 in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1]


def load_rows():
    rows = [json.loads(x) for x in (SH / "manifest.jsonl").open(encoding="utf-8") if x.strip()]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "validation"]
    return tr, va


def _feat(clip_id: str, dev):
    return torch.from_numpy(np.load(FEATS / f"{clip_id}.npy").astype(np.float32)).unsqueeze(0).to(dev)


@torch.no_grad()
def eval_fair(core, proj, bias, rows, dev, max_new: int = 448, limit: int = 300) -> dict:
    """Fair protocol on the PUBLIC validation split (subsampled for per-epoch speed; seeded)."""
    rng = random.Random(0)
    sample = rows if len(rows) <= limit else rng.sample(rows, limit)
    cns, wns = [], []
    for r in sample:
        fp = FEATS / f"{r['id']}.npy"
        if not fp.exists():
            continue
        audio = proj(_feat(r["id"], dev), structural_bias=bias)
        prefix = build_prefix(core, audio, "transcribe the speech")
        out = core.greedy_from_embeds(prefix, max_new=max_new, stop_token=EOS)
        pred = bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()
        p, g = _norm(pred), _norm(r["text"])
        cns.append(_lev(p, g) / max(1, len(g)))
        pw, gw = p.split(), g.split()
        wns.append(_lev(pw, gw) / max(1, len(gw)))
    return {"val_cer_norm_fair": round(float(np.mean(cns)), 4) if cns else None,
            "val_wer_norm_fair": round(float(np.mean(wns)), 4) if wns else None,
            "n_eval": len(cns)}


def main(epochs: int = 3, lr: float = 1e-4, r: int = 16, eos_weight: float = 3.0) -> int:
    t_start = time.time()
    core = Megatron1BCore().load()
    for p in core._model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias
    lora_params = inject_megatron_lora(core._model, r=r)

    d_enc = int(np.load(next(FEATS.glob("*.npy"))).shape[-1])
    proj = SphotaProjector(d_enc=d_enc, d_model=core.d_model, downsample=4).to(dev)

    # TRIZ P6 curriculum: warm-start projector + LoRA from the synthetic-Sanskrit XL winner
    warm = ROOT / "data/alm/xl1b_ckpt.pt"
    if warm.exists():
        blob = torch.load(warm, map_location=dev, weights_only=True)
        try:
            proj.load_state_dict(blob["projector"])
            n = load_megatron_lora(core._model, blob["lora"])
            print(json.dumps({"warm_start": "xl1b_ckpt.pt", "lora_tensors": n,
                              "from_val_cer": blob.get("val_cer_norm_fair")}), flush=True)
        except Exception as e:
            print(f"warm-start partial/skipped: {e}", flush=True)

    trainable = list(proj.parameters()) + lora_params
    opt = torch.optim.Adam(trainable, lr=lr)
    w = torch.ones(core.vocab_size, device=dev)
    w[EOS] = eos_weight
    ce = torch.nn.CrossEntropyLoss(weight=w)

    train_rows, val_rows = load_rows()
    train_rows = [r for r in train_rows if (FEATS / f"{r['id']}.npy").exists()]
    print(json.dumps({"n_train": len(train_rows), "n_val": len(val_rows)}), flush=True)

    def loss_on(row):
        audio = proj(_feat(row["id"], dev), structural_bias=bias)
        prefix = build_prefix(core, audio, "transcribe the speech")
        ids = torch.tensor([list(row["text"].encode("utf-8"))[:420] + [EOS]],
                           dtype=torch.long, device=dev)
        seq = torch.cat([prefix, core.embed_tokens(ids[:, :-1])], dim=1)
        logits = core.logits_from_embeds(seq)
        n = ids.shape[1]
        return ce(logits[:, -n:, :].reshape(-1, core.vocab_size), ids.reshape(-1))

    hist, fair_hist, best = [], [], float("inf")
    for ep in range(epochs):
        proj.train()
        order = list(range(len(train_rows)))
        random.Random(ep).shuffle(order)
        run, t0 = 0.0, time.time()
        for i, idx in enumerate(order):
            opt.zero_grad()
            L = loss_on(train_rows[idx])
            L.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            run += float(L.item())
            if (i + 1) % 1000 == 0:
                print(json.dumps({"epoch": ep, "step": i + 1, "of": len(order),
                                  "loss": round(run / (i + 1), 4),
                                  "min_elapsed": round((time.time() - t0) / 60, 1)}), flush=True)
        hist.append(round(run / len(order), 4))
        proj.eval()
        fair = eval_fair(core, proj, bias, val_rows, dev)
        fair_hist.append(fair)
        print(json.dumps({"epoch": ep, "train_loss": hist[-1], **fair}), flush=True)
        if fair["val_cer_norm_fair"] is not None and fair["val_cer_norm_fair"] < best:
            best = fair["val_cer_norm_fair"]
            torch.save({"projector": proj.state_dict(), "lora": megatron_lora_state_dict(core._model),
                        "d_enc": d_enc, "d_model": core.d_model, "r": r, "epoch": ep,
                        "val_cer_norm_fair": best},
                       ROOT / "data/alm/sh1b_ckpt.pt")
            print(json.dumps({"checkpoint_saved": ep, "best_val_cer_norm": best}), flush=True)

    metrics = {"method": f"Shrutilipi-sa adaptation (1.13B + MegatronLoRA r={r}, warm-start xl1b, "
                          f"eos_weight={eos_weight}, grad_clip=1.0, lr={lr})",
               "corpus": "amithm3/shrutilipi:sa (REAL human Sanskrit; public splits)",
               "epochs": epochs, "train_loss_history": hist, "fair_eval_history": fair_hist,
               "protocol": "free greedy, max_new=448, stop at model EOS; val split only — TEST untouched",
               "wall_hours": round((time.time() - t_start) / 3600, 2)}
    (ROOT / "data/alm/sh1b_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
