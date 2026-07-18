"""XL retrain: projector + LoRA on the expanded native-Sanskrit corpus, EOS-weighted, fairly evaluated.

Fixes the three honest deficits the corrected AA benchmark exposed:
  1. DATA — trains on speech_corpus_indic_xl (~10k clips, 16× v1) built from PSALM's full gold-kāraka
     fixture; v1's 58-clip val is FROZEN so results stay comparable.
  2. EOS — every target ends in the EOS sentinel and the CE loss up-weights that class (the lora_ckpt
     never learned to stop → free-decode CER 1.82). Instruction-style multi-task (transcribe + kāraka
     + language) like train_instruct.py.
  3. EVAL — the reported metric is the FAIR protocol: free greedy decode, fixed 64-byte budget, stop
     at the model's own EOS, scored as transliteration-normalized CER (same fold as _rescore_alm.py).
     No gold-length oracle anywhere.

Writes phase-specific artifacts (xl_ckpt.pt / xl_metrics.json) — never clobbers earlier phases'
evidence (the gate_LR collision class of bug).

Run in container:  scripts/alm/in_container.sh python /work/pranava/scripts/alm/train_xl.py [epochs]
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

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.instruct import EOS, build_prefix, tasks_for
from pranava.alm.lora import inject_lora, lora_state_dict
from pranava.alm.projector import SphotaProjector

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
XL = ROOT / "data/alm/speech_corpus_indic_xl"
FEATS = XL / "feats"

# ---- fair normalized-CER metric (mirror of scripts/alm/_rescore_alm.py, SLP1 side) -----------------
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


def cer_norm(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    return _lev(p, g) / max(1, len(g))


# ---- data ------------------------------------------------------------------------------------------
def load_rows() -> tuple[list[dict], list[dict]]:
    rows = [json.loads(x) for x in (XL / "manifest.jsonl").open(encoding="utf-8") if x.strip()]
    return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]


def _feat(clip_id: str, dev):
    return torch.from_numpy(np.load(FEATS / f"{clip_id}.npy").astype(np.float32)).unsqueeze(0).to(dev)


def _logits(core, seq):
    x = seq
    for b in core.model.blocks:
        x = b(x)
    return core.model.head(core.model.norm_f(x))


@torch.no_grad()
def eval_fair(core, proj, bias, val_rows, dev) -> dict:
    """Frozen-val transcription under the FAIR protocol → normalized + raw CER."""
    cns, crs = [], []
    for r in val_rows:
        audio = proj(_feat(r["id"], dev), structural_bias=bias)
        prefix = build_prefix(core, audio, "transcribe the speech")
        out = core.greedy_from_embeds(prefix, max_new=64, stop_token=EOS)
        pred = bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()
        cns.append(cer_norm(pred, r["text"]))
        crs.append(_lev(pred, r["text"]) / max(1, len(r["text"])))
    return {"val_cer_norm_fair": round(float(np.mean(cns)), 4),
            "val_cer_raw_fair": round(float(np.mean(crs)), 4)}


def main(epochs: int = 3, lr: float = 3e-4, r: int = 8, eos_weight: float = 3.0,
         per_epoch_extractive: int = 12000) -> int:
    t_start = time.time()
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias
    lora_params = inject_lora(core.model, r=r, alpha=2 * r)
    core.model.to(dev)

    d_enc = int(np.load(next(FEATS.glob("*.npy"))).shape[-1])
    proj = SphotaProjector(d_enc=d_enc, d_model=core.d_model, downsample=4).to(dev)

    # warm-start from the instruct checkpoint (best current EOS behaviour)
    warm = ROOT / "data/alm/instruct_ckpt.pt"
    if warm.exists():
        blob = torch.load(warm, map_location=dev, weights_only=True)
        try:
            proj.load_state_dict(blob["projector"])
            sd = {n: p for n, p in core.model.named_parameters() if ".A" in n or ".B" in n}
            for n, w in blob.get("lora", {}).items():
                if n in sd:
                    sd[n].data.copy_(w.to(dev))
            print("warm-started from instruct_ckpt.pt", flush=True)
        except Exception as e:  # shape mismatch (different encoder) → cold start, honestly reported
            print(f"warm-start skipped: {e}", flush=True)

    opt = torch.optim.Adam(list(proj.parameters()) + lora_params, lr=lr)
    w = torch.ones(core.vocab_size, device=dev)
    w[EOS] = eos_weight  # make 'stop' as learnable as content — the anti-ramble fix
    ce = torch.nn.CrossEntropyLoss(weight=w)

    train_rows, val_rows = load_rows()
    print(json.dumps({"n_train_clips": len(train_rows), "n_val": len(val_rows)}), flush=True)

    # instruction examples: ALL transcribe items every epoch + a rotating sample of extractive tasks
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
        logits = _logits(core, seq)
        n = ids.shape[1]
        return ce(logits[:, -n:, :].reshape(-1, core.vocab_size), ids.reshape(-1))

    hist, fair_hist = [], []
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
            if (i + 1) % 2000 == 0:
                print(json.dumps({"epoch": ep, "step": i + 1, "of": len(batch),
                                  "loss": round(run / (i + 1), 4),
                                  "min_elapsed": round((time.time() - t0) / 60, 1)}), flush=True)
        hist.append(round(run / len(batch), 4))
        proj.eval()
        fair = eval_fair(core, proj, bias, val_rows, dev)
        fair_hist.append(fair)
        print(json.dumps({"epoch": ep, "train_loss": hist[-1], **fair}), flush=True)
        # keep the BEST checkpoint by fair val CER (1B smoke showed per-epoch overfitting on small data)
        if not fair_hist[:-1] or fair["val_cer_norm_fair"] < min(f["val_cer_norm_fair"] for f in fair_hist[:-1]):
            torch.save({"projector": proj.state_dict(), "lora": lora_state_dict(core.model),
                        "d_enc": d_enc, "d_model": core.d_model, "r": r, "epoch": ep,
                        "val_cer_norm_fair": fair["val_cer_norm_fair"]},
                       ROOT / "data/alm/xl_ckpt.pt")
            print(json.dumps({"checkpoint_saved": ep, "best_cer_norm": fair["val_cer_norm_fair"]}), flush=True)

    metrics = {"method": f"XL instruct SFT (projector+LoRA r={r}, eos_weight={eos_weight})",
               "corpus": "speech_corpus_indic_xl", "epochs": epochs,
               "n_train_clips": len(train_rows), "train_loss_history": hist,
               "fair_eval_history": fair_hist, **fair_hist[-1],
               "protocol": "free greedy decode, 64-byte budget, stop at model EOS — no gold-length oracle",
               "wall_hours": round((time.time() - t_start) / 3600, 2)}
    (ROOT / "data/alm/xl_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
