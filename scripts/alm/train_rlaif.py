"""RLAIF: align the instruction-tuned Śabda-ALM with an AI-feedback reward via DPO.

From the SFT model we sample K candidate responses per (audio, instruction), score each with an
automated **AI-feedback reward** — task-correctness against the gold field plus conciseness, which
penalises the run-on over-generation the raw model sometimes produces — and form (chosen, rejected)
preference pairs. Then Direct Preference Optimization raises the policy's log-prob margin for the
chosen response over the rejected one, relative to a frozen reference (the SFT model itself). Only
projector + LoRA update. Reports per-task exact-match vs the SFT baseline. Run in prabhasa/nemo-gb10.

    python scripts/alm/train_rlaif.py [n_prompts] [dpo_epochs]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.instruct import EOS, build_prefix
from pranava.alm.lora import inject_lora, lora_state_dict
from pranava.alm.projector import SphotaProjector
from pranava.alm.train import _levenshtein

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[1]
FEATS = ROOT / "data/alm/speech_corpus_multi/feats"
CORPUS = ROOT / "data/alm/instruct_corpus/manifest.jsonl"


def _feat(wav, dev):
    return torch.from_numpy(np.load(FEATS / f"{Path(wav).stem}.npy").astype(np.float32)).unsqueeze(0).to(dev)


def _logits(core, seq):
    x = seq
    for b in core.model.blocks:
        x = b(x)
    return core.model.head(core.model.norm_f(x))


def ai_reward(cand: str, gold: str, task: str) -> float:
    """Automated AI-feedback reward: correctness (edit-similarity to the gold field) + conciseness
    (penalise over-generation). Conciseness matters most for the long transcribe task, where the raw
    model rambles; for one-word kāraka answers correctness dominates."""
    c, g = list(cand.encode()), list(gold.encode())
    sim = 1.0 - _levenshtein(c, g) / max(1, max(len(c), len(g)))
    over = max(0, len(c) - len(g)) / max(1, len(g))
    concise = max(0.0, 1.0 - 0.5 * over)
    return 0.8 * sim + 0.2 * concise


def _sample(core, prefix, max_new, temp, dev):
    x = prefix; out = []
    for _ in range(max_new):
        logits = core.forward_embeds(x)[0, -1] / temp
        p = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(p, 1).item())
        if nxt == EOS:
            break
        out.append(nxt)
        x = torch.cat([x, core.model.embed(torch.tensor([[nxt]], device=dev))], dim=1)
    return bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()


def _seq_logprob(core, prefix, resp: str, dev):
    ids = torch.tensor([list(resp.encode("utf-8")) + [EOS]], dtype=torch.long, device=dev)
    seq = torch.cat([prefix, core.embed_tokens(ids[:, :-1])], dim=1)
    logits = _logits(core, seq)
    n = ids.shape[1]
    lp = F.log_softmax(logits[:, -n:, :], dim=-1)
    return lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1).sum()


@torch.no_grad()
def _accuracy(core, proj, bias, val, dev):
    from collections import defaultdict
    hit = defaultdict(int); tot = defaultdict(int)
    for ex in val:
        audio = proj(_feat(ex["wav"], dev), structural_bias=bias)
        prefix = build_prefix(core, audio, ex["instruction"])
        out = core.greedy_from_embeds(prefix, max_new=64, stop_token=EOS)
        pred = bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()
        tot[ex["task"]] += 1; hit[ex["task"]] += int(pred == ex["response"])
    return {t: round(hit[t] / tot[t], 4) for t in sorted(tot)}, \
        round(sum(hit.values()) / max(1, sum(tot.values())), 4)


def main(n_prompts: int = 300, dpo_epochs: int = 2, beta: float = 0.3, k: int = 4,
         per_task_cap: int = 30, lr: float = 5e-5) -> int:
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias
    lora_params = inject_lora(core.model, r=8, alpha=16)
    core.model.to(dev)
    d_enc = int(np.load(next(FEATS.glob("*.npy"))).shape[-1])
    proj = SphotaProjector(d_enc=d_enc, d_model=core.d_model, downsample=4).to(dev)

    sft = ROOT / "data/alm/instruct_ckpt.pt"
    blob = torch.load(sft, map_location=dev, weights_only=True)
    proj.load_state_dict(blob["projector"])
    lsd = {n: p for n, p in core.model.named_parameters() if ".A" in n or ".B" in n}
    for n, w in blob["lora"].items():
        if n in lsd:
            lsd[n].data.copy_(w.to(dev))

    rows = [json.loads(l) for l in CORPUS.open()]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    torch.manual_seed(0)
    prompts = [train[int(i)] for i in torch.randperm(len(train))[:n_prompts]]

    base_per_task, base_overall = _accuracy(core, proj, bias, val, dev)

    # --- build preference pairs with the AI-feedback reward (sampling under the SFT policy) -------
    raw = []
    proj.eval()
    with torch.no_grad():
        for ex in prompts:
            audio = proj(_feat(ex["wav"], dev), structural_bias=bias)
            prefix = build_prefix(core, audio, ex["instruction"])
            mn = len(ex["response"].encode()) + 8
            cands = {_sample(core, prefix, mn, 0.9, dev) for _ in range(k)}
            cands.add(ex["response"])  # ensure the gold is in the pool
            scored = sorted(cands, key=lambda c: ai_reward(c, ex["response"], ex["task"]))
            if ai_reward(scored[-1], ex["response"], ex["task"]) - \
               ai_reward(scored[0], ex["response"], ex["task"]) > 0.05:
                raw.append((ex, scored[-1], scored[0]))  # (prompt, chosen, rejected)

    # Balance pairs per task so the long, high-gradient `transcribe` pairs don't dominate the shared
    # LoRA update and corrupt the well-calibrated extractive tasks (the β=0.1 regression we saw).
    from collections import defaultdict
    cap = per_task_cap
    buckets: dict[str, list] = defaultdict(list)
    for p in raw:
        buckets[p[0]["task"]].append(p)
    pairs = [p for b in buckets.values() for p in b[:cap]]
    print(json.dumps({"n_pairs_raw": len(raw), "n_pairs_balanced": len(pairs),
                      "per_task": {t: min(len(b), cap) for t, b in buckets.items()}}), flush=True)

    # --- precompute reference (SFT) log-probs; then DPO on the policy ----------------------------
    ref = {}
    with torch.no_grad():
        for idx, (ex, ch, rj) in enumerate(pairs):
            audio = proj(_feat(ex["wav"], dev), structural_bias=bias)
            prefix = build_prefix(core, audio, ex["instruction"])
            ref[idx] = (_seq_logprob(core, prefix, ch, dev).item(),
                        _seq_logprob(core, prefix, rj, dev).item())

    opt = torch.optim.Adam(list(proj.parameters()) + lora_params, lr=lr)
    hist = []
    for ep in range(dpo_epochs):
        proj.train(); run = 0.0
        for idx in torch.randperm(len(pairs)):
            ex, ch, rj = pairs[int(idx)]
            audio = proj(_feat(ex["wav"], dev), structural_bias=bias)
            prefix = build_prefix(core, audio, ex["instruction"])
            lp_c = _seq_logprob(core, prefix, ch, dev)
            lp_r = _seq_logprob(core, prefix, rj, dev)
            rc, rr = ref[int(idx)]
            margin = (lp_c - rc) - (lp_r - rr)
            loss = -F.logsigmoid(beta * margin)
            opt.zero_grad(); loss.backward(); opt.step(); run += float(loss.item())
        hist.append(round(run / max(1, len(pairs)), 4))
        print(json.dumps({"dpo_epoch": ep, "loss": hist[-1]}), flush=True)

    proj.eval()
    rlaif_per_task, rlaif_overall = _accuracy(core, proj, bias, val, dev)
    torch.save({"projector": proj.state_dict(), "lora": lora_state_dict(core.model),
                "d_enc": d_enc, "d_model": core.d_model, "r": 8},
               ROOT / "data/alm/rlaif_ckpt.pt")
    out = {"method": "RLAIF via DPO (AI-feedback reward: correctness + conciseness) over SFT",
           "reward": "0.8*edit_similarity_to_gold + 0.2*conciseness", "beta": beta, "k_samples": k,
           "per_task_cap": per_task_cap, "lr": lr,
           "n_prompts": n_prompts, "n_preference_pairs": len(pairs), "dpo_epochs": dpo_epochs,
           "dpo_loss_history": hist,
           "sft_overall_accuracy": base_overall, "rlaif_overall_accuracy": rlaif_overall,
           "sft_per_task": base_per_task, "rlaif_per_task": rlaif_per_task,
           "improves_on_sft": bool(rlaif_overall >= base_overall)}
    (ROOT / "data/alm/rlaif_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    a = sys.argv
    raise SystemExit(main(int(a[1]) if len(a) > 1 else 300,
                          int(a[2]) if len(a) > 2 else 2,
                          float(a[3]) if len(a) > 3 else 0.3))
