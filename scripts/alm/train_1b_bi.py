"""BILINGUAL public-benchmark training: one Śabda-ALM for English (LibriSpeech) + Sanskrit (Shrutilipi+Vagdhenu).

The operator's hypothesis: the sphoṭa-principled byte core (prabhasa English↔Sanskrit architecture)
must stand head-to-head on the baselines' home language, not just its own. One model, two languages,
zero per-language branches: English targets are lowercase ASCII, Sanskrit targets SLP1 — the byte
vocabulary carries both natively; the audio decides the language.

Curriculum: warm-start projector+LoRA from sh1b_ckpt.pt (real-Sanskrit adapted) if present, else
xl1b_ckpt.pt. Per-epoch fair eval on BOTH public validation sets (LS dev-clean subsample + Shrutilipi
validation subsample); best checkpoint by the MACRO-AVERAGE of the two val CERs (bilingual balance —
optimizing either language alone is not the goal). Tests remain untouched.

Artifacts: data/alm/bi1b_ckpt.pt, data/alm/bi1b_metrics.json.
Run on the RTX 5090: python /work/pranava/scripts/alm/train_1b_bi.py [epochs]
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
SETS = {
    "librispeech": ROOT / "data/alm/librispeech",
    "shrutilipi": ROOT / "data/alm/shrutilipi_sa",
    "vagdhenu": ROOT / "data/alm/vagdhenu",
}

_SLP1_IAST = {"A": "ā", "I": "ī", "U": "ū", "f": "ṛ", "F": "ṝ", "x": "ḷ", "X": "ḹ", "E": "ai",
              "O": "au", "M": "ṃ", "H": "ḥ", "K": "kh", "G": "gh", "N": "ṅ", "C": "ch", "J": "jh",
              "Y": "ñ", "w": "ṭ", "W": "ṭh", "q": "ḍ", "Q": "ḍh", "R": "ṇ", "T": "th", "D": "dh",
              "P": "ph", "B": "bh", "S": "ś", "z": "ṣ", "L": "ḷ"}


def _fold(s: str, slp1: bool) -> str:
    if slp1:
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


def load_split(name: str, split: str) -> list[dict]:
    p = SETS[name] / "manifest.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]
    out = [r for r in rows if r["split"] == split]
    for r in out:
        r["_set"] = name
        r["_slp1"] = name != "librispeech"
    return out


def _feat(row, dev):
    p = SETS[row["_set"]] / "feats" / f"{row['id']}.npy"
    return torch.from_numpy(np.load(p).astype(np.float32)).unsqueeze(0).to(dev) if p.exists() else None


@torch.no_grad()
def eval_fair(core, proj, bias, rows, dev, slp1: bool, max_new: int = 448, limit: int = 200) -> dict:
    rng = random.Random(0)
    sample = rows if len(rows) <= limit else rng.sample(rows, limit)
    cns, wns = [], []
    for r in sample:
        t = _feat(r, dev)
        if t is None:
            continue
        audio = proj(t, structural_bias=bias)
        prefix = build_prefix(core, audio, "transcribe the speech")
        out = core.greedy_from_embeds(prefix, max_new=max_new, stop_token=EOS, no_repeat_ngram=6)
        pred = bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()
        p, g = _fold(pred, slp1), _fold(r["text"], slp1)
        cns.append(_lev(p, g) / max(1, len(g)))
        pw, gw = p.split(), g.split()
        wns.append(_lev(pw, gw) / max(1, len(gw)))
    return {"cer": round(float(np.mean(cns)), 4) if cns else None,
            "wer": round(float(np.mean(wns)), 4) if wns else None, "n": len(cns)}


def main(epochs: int = 2, lr: float = 1e-4, r: int = 64, eos_weight: float = 3.0) -> int:
    t_start = time.time()
    core = Megatron1BCore().load()
    for p in core._model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias
    lora_params = inject_megatron_lora(core._model, r=r)

    # pick a feats dir that actually CONTAINS .npy files (an empty dir crashed the first launch)
    first_npy = next((f for s in SETS for f in sorted((SETS[s] / "feats").glob("*.npy"))[:1]
                      if (SETS[s] / "feats").exists()), None)
    if first_npy is None:
        raise SystemExit("no feats found in any corpus — run precompute_feats_dir.py first")
    d_enc = int(np.load(first_npy).shape[-1])
    proj = SphotaProjector(d_enc=d_enc, d_model=core.d_model, downsample=4).to(dev)

    warm = next((p for p in (ROOT / "data/alm/bi1b_ckpt.pt", ROOT / "data/alm/sh1b_ckpt.pt", ROOT / "data/alm/xl1b_ckpt.pt") if p.exists()), None)
    if warm:
        blob = torch.load(warm, map_location=dev, weights_only=True)
        try:
            proj.load_state_dict(blob["projector"])
            n = load_megatron_lora(core._model, blob["lora"])
            print(json.dumps({"warm_start": warm.name, "lora_tensors": n}), flush=True)
        except Exception as e:
            print(f"warm-start skipped: {e}", flush=True)

    trainable = list(proj.parameters()) + lora_params
    opt = torch.optim.Adam(trainable, lr=lr)
    w = torch.ones(core.vocab_size, device=dev)
    w[EOS] = eos_weight
    ce = torch.nn.CrossEntropyLoss(weight=w)

    train_rows = []
    for s in SETS:
        rows = [r_ for r_ in load_split(s, "train")
                if (SETS[s] / "feats" / f"{r_['id']}.npy").exists()]
        train_rows += rows * (2 if s == "librispeech" else 1)  # EN-weighted: audio-conditioning for
        print(json.dumps({"set": s, "n_train_with_feats": len(rows),   # English needs more gradient
                          "weight": 2 if s == "librispeech" else 1}), flush=True)
    val_en = load_split("librispeech", "validation")
    val_sa = load_split("shrutilipi", "validation")

    def loss_on(row):
        t = _feat(row, dev)
        audio = proj(t, structural_bias=bias)
        prefix = build_prefix(core, audio, "transcribe the speech")
        ids = torch.tensor([list(row["text"].encode("utf-8"))[:420] + [EOS]], dtype=torch.long, device=dev)
        seq = torch.cat([prefix, core.embed_tokens(ids[:, :-1])], dim=1)
        logits = core.logits_from_embeds(seq)
        n = ids.shape[1]
        return ce(logits[:, -n:, :].reshape(-1, core.vocab_size), ids.reshape(-1))

    hist, evals, best = [], [], float("inf")
    for ep in range(epochs):
        for g in opt.param_groups:  # per-epoch decay: 1e-4 -> xN(0.6) — epoch-1 collapse fix
            g["lr"] = lr * (0.6 ** ep)
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
            if (i + 1) % 2000 == 0:
                print(json.dumps({"epoch": ep, "step": i + 1, "of": len(order),
                                  "loss": round(run / (i + 1), 4),
                                  "min_elapsed": round((time.time() - t0) / 60, 1)}), flush=True)
        hist.append(round(run / len(order), 4))
        proj.eval()
        en = eval_fair(core, proj, bias, val_en, dev, slp1=False)
        sa = eval_fair(core, proj, bias, val_sa, dev, slp1=True)
        macro = round((en["cer"] + sa["cer"]) / 2, 4) if en["cer"] is not None and sa["cer"] is not None else None
        evals.append({"epoch": ep, "en_val": en, "sa_val": sa, "macro_cer": macro})
        print(json.dumps(evals[-1] | {"train_loss": hist[-1]}), flush=True)
        if macro is not None and macro < best:
            best = macro
            torch.save({"projector": proj.state_dict(), "lora": megatron_lora_state_dict(core._model),
                        "d_enc": d_enc, "d_model": core.d_model, "r": r, "epoch": ep,
                        "val_cer_norm_fair": best, "en_val": en, "sa_val": sa},
                       ROOT / "data/alm/bi1b_ckpt.pt")
            print(json.dumps({"checkpoint_saved": ep, "best_macro_cer": best}), flush=True)

    metrics = {"method": f"BILINGUAL public SFT (1.13B+LoRA r={r}, warm {warm.name if warm else 'none'}, "
                          f"eos_w={eos_weight}, clip 1.0, lr {lr})",
               "corpora": "LibriSpeech train-clean-100 (en) + Shrutilipi-sa train + Vagdhenu train (sa)",
               "epochs": epochs, "train_loss_history": hist, "eval_history": evals,
               "selection": "best checkpoint by MACRO-AVG of en+sa val CER (bilingual balance)",
               "protocol": "free greedy, max_new 448, model EOS; public tests untouched",
               "wall_hours": round((time.time() - t_start) / 3600, 2)}
    (ROOT / "data/alm/bi1b_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2))
