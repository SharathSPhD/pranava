"""1.13B XL: projector + Megatron-LoRA on the expanded corpus — the scale arm of the leaderboard push.

Same recipe as scripts/alm/train_xl.py (instruct-style multi-task, EOS-weighted loss, fair free-decode
eval on the frozen 58-clip val) but on the fully-trained 1.13B Megatron core (m4/final.pt) with LoRA
injected into its MLP + attention linears (src/pranava/alm/megatron_lora.py; Mamba mixers untouched).
P5 showed scale alone helps (projector-only 0.571 vs 0.774); this adds adaptation + data + EOS.

Artifacts: data/alm/xl1b_ckpt.pt, data/alm/xl1b_metrics.json (phase-specific — no clobbering).
Run on the RTX 5090 in prabhasa/nemo-5090:26.02 (cwd /work):
  python /work/pranava/scripts/alm/train_1b_xl.py [epochs]
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

from pranava.alm.instruct import EOS, build_prefix, tasks_for
from pranava.alm.megatron_core import Megatron1BCore
from pranava.alm.megatron_lora import inject_megatron_lora, megatron_lora_state_dict
from pranava.alm.projector import SphotaProjector

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
XL = ROOT / "data/alm/speech_corpus_indic_xl"
FEATS = XL / "feats"

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
    rows = [json.loads(x) for x in (XL / "manifest.jsonl").open(encoding="utf-8") if x.strip()]
    return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]


def _feat(clip_id: str, dev):
    return torch.from_numpy(np.load(FEATS / f"{clip_id}.npy").astype(np.float32)).unsqueeze(0).to(dev)


@torch.no_grad()
def eval_fair(core, proj, bias, val_rows, dev) -> dict:
    cns, crs = [], []
    for r in val_rows:
        audio = proj(_feat(r["id"], dev), structural_bias=bias)
        prefix = build_prefix(core, audio, "transcribe the speech")
        out = core.greedy_from_embeds(prefix, max_new=64, stop_token=EOS)
        pred = bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()
        p, g = _norm(pred), _norm(r["text"])
        cns.append(_lev(p, g) / max(1, len(g)))
        crs.append(_lev(pred, r["text"]) / max(1, len(r["text"])))
    return {"val_cer_norm_fair": round(float(np.mean(cns)), 4),
            "val_cer_raw_fair": round(float(np.mean(crs)), 4)}


def main(epochs: int = 2, lr: float = 2e-4, r: int = 16, eos_weight: float = 3.0,
         per_epoch_extractive: int = 8000) -> int:
    t_start = time.time()
    core = Megatron1BCore().load()
    for p in core._model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias
    lora_params = inject_megatron_lora(core._model, r=r)
    n_lora = sum(p.numel() for p in lora_params)
    print(json.dumps({"model": "1.13B m4/final.pt", "n_lora_params": n_lora, "r": r}), flush=True)

    d_enc = int(np.load(next(FEATS.glob("*.npy"))).shape[-1])
    proj = SphotaProjector(d_enc=d_enc, d_model=core.d_model, downsample=4).to(dev)
    # warm-start projector from P5's 1B run if compatible (it learned audio→1B alignment already)
    warm = ROOT / "data/alm/projector_1b_multi.pt"
    if not warm.exists():
        warm = ROOT / "data/alm/projector_1b.pt"
    if warm.exists():
        try:
            blob = torch.load(warm, map_location=dev, weights_only=True)
            proj.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob["projector"])
            print(f"projector warm-started from {warm.name}", flush=True)
        except Exception as e:
            print(f"projector warm-start skipped: {e}", flush=True)

    opt = torch.optim.Adam(list(proj.parameters()) + lora_params, lr=lr)
    w = torch.ones(core.vocab_size, device=dev)
    w[EOS] = eos_weight
    ce = torch.nn.CrossEntropyLoss(weight=w)

    train_rows, val_rows = load_rows()
    print(json.dumps({"n_train_clips": len(train_rows), "n_val": len(val_rows)}), flush=True)

    def epoch_examples(seed: int):
        transcribe, extractive = [], []
        for row in train_rows:
            for ex in tasks_for(wav=row["id"], text=row["text"], karaka=row.get("karaka", []),
                                lang="sa", split="train"):
                (transcribe if ex.task == "transcribe" else extractive).append(ex)
        rng = random.Random(seed)
        rng.shuffle(extractive)
        batch = transcribe + extractive[:per_epoch_extractive]
        rng.shuffle(batch)
        return batch

    def loss_on(ex):
        audio = proj(_feat(ex.wav, dev), structural_bias=bias)
        prefix = build_prefix(core, audio, ex.instruction)
        ids = torch.tensor([list(ex.response.encode("utf-8")) + [EOS]], dtype=torch.long, device=dev)
        seq = torch.cat([prefix, core.embed_tokens(ids[:, :-1])], dim=1)
        logits = core.logits_from_embeds(seq)
        n = ids.shape[1]
        return ce(logits[:, -n:, :].reshape(-1, core.vocab_size), ids.reshape(-1))

    hist, fair_hist = [], []
    best = float("inf")
    for ep in range(epochs):
        proj.train()
        batch = epoch_examples(seed=ep)
        run, t0 = 0.0, time.time()
        for i, ex in enumerate(batch):
            opt.zero_grad()
            L = loss_on(ex)
            L.backward()
            opt.step()
            run += float(L.item())
            if (i + 1) % 1000 == 0:
                print(json.dumps({"epoch": ep, "step": i + 1, "of": len(batch),
                                  "loss": round(run / (i + 1), 4),
                                  "min_elapsed": round((time.time() - t0) / 60, 1)}), flush=True)
        hist.append(round(run / len(batch), 4))
        proj.eval()
        fair = eval_fair(core, proj, bias, val_rows, dev)
        fair_hist.append(fair)
        print(json.dumps({"epoch": ep, "train_loss": hist[-1], **fair}), flush=True)
        # keep the BEST checkpoint by fair val CER — the smoke run showed tiny-data epochs overfit
        if fair["val_cer_norm_fair"] < best:
            best = fair["val_cer_norm_fair"]
            torch.save({"projector": proj.state_dict(), "lora": megatron_lora_state_dict(core._model),
                        "d_enc": d_enc, "d_model": core.d_model, "r": r, "epoch": ep,
                        "val_cer_norm_fair": best},
                       ROOT / "data/alm/xl1b_ckpt.pt")
            print(json.dumps({"checkpoint_saved": ep, "best_cer_norm": best}), flush=True)

    metrics = {"method": f"1.13B XL instruct SFT (projector+MegatronLoRA r={r}, eos_weight={eos_weight})",
               "corpus": "speech_corpus_indic_xl", "epochs": epochs, "n_lora_params": n_lora,
               "n_train_clips": len(train_rows), "train_loss_history": hist,
               "fair_eval_history": fair_hist, **(fair_hist[-1] if fair_hist else {}),
               "protocol": "free greedy decode, 64-byte budget, stop at model EOS — no gold-length oracle",
               "wall_hours": round((time.time() - t_start) / 3600, 2)}
    (ROOT / "data/alm/xl1b_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2))
