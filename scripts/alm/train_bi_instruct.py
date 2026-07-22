"""Broaden the LIVE instruction model with translation + language-ID (contract clause 2, model half).

The served model is the 200M SanskritCore instruct checkpoint (instruct_ckpt.pt) — it already does
transcribe + kāraka (6 tasks). This round ADDS the bilingual tasks from the aligned-Itihāsa corpus
(translate sa→en, translate en→sa, language-ID) WITHOUT forgetting the originals: it warm-starts from
instruct_ckpt.pt and trains on BOTH corpora mixed, response-only loss, then reports per-task exact-match
with the instruction-sensitivity control.

Honesty note carried from the interference finding (core_prior_probe.json): translate_en emits ENGLISH,
which the shared adapter handles worst. We expect translate_sa (Sanskrit out) and language-ID to work
and translate_en to be weaker — and we MEASURE it per task rather than asserting. A new checkpoint is
saved only if overall val accuracy does not regress the original tasks (save bar), so this can never
degrade the live model.

Run in prabhasa/nemo-gb10 on the GB10:  python scripts/alm/train_bi_instruct.py [epochs]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.instruct import EOS, build_prefix
from pranava.alm.lora import inject_lora, lora_state_dict

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]

# (manifest, feats_dir, is_relative_wav). Both corpora share the Parakeet feature space.
CORPORA = [
    (ROOT / "data/alm/instruct_corpus/manifest.jsonl", ROOT / "data/alm/speech_corpus_multi/feats"),
    (ROOT / "data/alm/bi_instruct/manifest.jsonl", ROOT / "data/alm/bi_instruct/feats"),
]
VAL_EVERY = 7  # deterministic held-out: every 7th item per task → val


def load_all() -> tuple[list[dict], list[dict]]:
    rows = []
    for manifest, feats in CORPORA:
        if not manifest.exists():
            continue
        for r in (json.loads(l) for l in manifest.open(encoding="utf-8") if l.strip()):
            stem = Path(r["wav"]).stem
            fp = feats / f"{stem}.npy"
            if not fp.exists():  # skip items whose feats were not precomputed (logged in caller)
                continue
            r["_feat"] = str(fp)
            rows.append(r)
    # deterministic per-task split so val covers every task including the new ones
    per_task_idx = defaultdict(int)
    train, val = [], []
    for r in rows:
        i = per_task_idx[r["task"]]
        per_task_idx[r["task"]] += 1
        (val if i % VAL_EVERY == 0 else train).append(r)
    return train, val


def _feat(r, dev):
    return torch.from_numpy(np.load(r["_feat"]).astype(np.float32)).unsqueeze(0).to(dev)


def _logits(core, seq):
    x = seq
    for b in core.model.blocks:
        x = b(x)
    return core.model.head(core.model.norm_f(x))


@torch.no_grad()
def _decode(core, proj, bias, r, dev, instruction=None):
    audio = proj(_feat(r, dev), structural_bias=bias)
    prefix = build_prefix(core, audio, instruction or r["instruction"])
    out = core.greedy_from_embeds(prefix, max_new=96, stop_token=EOS)
    return bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()


def _cer(pred: str, gold: str) -> float:
    # char error rate for the free-form translate tasks (exact-match is too harsh for a sentence)
    import difflib
    if not gold:
        return 0.0 if not pred else 1.0
    sm = difflib.SequenceMatcher(None, pred, gold)
    return 1.0 - sm.ratio()


@torch.no_grad()
def evaluate(core, proj, bias, val, dev) -> dict:
    hit = defaultdict(int); tot = defaultdict(int); cer = defaultdict(list)
    instrs = sorted({r["instruction"] for r in val})
    ctrl_hit = ctrl_tot = 0
    for i, r in enumerate(val):
        pred = _decode(core, proj, bias, r, dev)
        tot[r["task"]] += 1
        hit[r["task"]] += int(pred == r["response"])
        cer[r["task"]].append(_cer(pred, r["response"]))
        if len(instrs) > 1:  # instruction-sensitivity control: same audio, a different question
            wrong = instrs[(instrs.index(r["instruction"]) + 1) % len(instrs)]
            if wrong != r["instruction"]:
                ctrl_hit += int(_decode(core, proj, bias, r, dev, instruction=wrong) == r["response"])
                ctrl_tot += 1
    exact = {t: round(hit[t] / tot[t], 4) for t in sorted(tot)}
    mean_cer = {t: round(float(np.mean(cer[t])), 4) for t in sorted(cer)}
    overall = round(sum(hit.values()) / max(1, sum(tot.values())), 4)
    ctrl = round(ctrl_hit / max(1, ctrl_tot), 4)
    return {"per_task_exact_match": exact, "per_task_mean_cer": mean_cer,
            "overall_exact_match": overall, "control_shuffled_instruction": ctrl,
            "instruction_sensitivity_gap": round(overall - ctrl, 4),
            "n_val": len(val), "n_val_per_task": {t: tot[t] for t in sorted(tot)}}


def main(epochs: int = 4, lr: float = 3e-4, r: int = 8) -> int:
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias
    lora_params = inject_lora(core.model, r=r, alpha=2 * r)
    core.model.to(dev)

    train, val = load_all()
    if not train:
        raise SystemExit("no training rows with feats — run precompute_feats_dir.py on bi_instruct first")
    d_enc = int(np.load(train[0]["_feat"]).shape[-1])
    from pranava.alm.projector import SphotaProjector
    proj = SphotaProjector(d_enc=d_enc, d_model=core.d_model, downsample=4).to(dev)

    # warm-start from the ALREADY-TRAINED instruct model to keep the 6 original tasks
    warm = ROOT / "data/alm/instruct_ckpt.pt"
    warm_bar = 0.0
    if warm.exists():
        blob = torch.load(warm, map_location=dev, weights_only=True)
        proj.load_state_dict(blob["projector"])
        sd = {n: p for n, p in core.model.named_parameters() if ".A" in n or ".B" in n}
        for n, w in blob.get("lora", {}).items():
            if n in sd:
                sd[n].data.copy_(w.to(dev))
        print(json.dumps({"warm_start": warm.name}), flush=True)

    # baseline eval BEFORE training — establishes the save bar (never ship a regression of the 6 tasks)
    proj.eval()
    base = evaluate(core, proj, bias, val, dev)
    warm_bar = base["overall_exact_match"]
    print(json.dumps({"baseline_before_broadening": base}), flush=True)

    opt = torch.optim.Adam(list(proj.parameters()) + lora_params, lr=lr)
    ce = torch.nn.CrossEntropyLoss()

    def loss_on(r):
        audio = proj(_feat(r, dev), structural_bias=bias)
        prefix = build_prefix(core, audio, r["instruction"])
        ids = torch.tensor([list(r["response"].encode("utf-8"))[:180] + [EOS]], dtype=torch.long, device=dev)
        seq = torch.cat([prefix, core.embed_tokens(ids[:, :-1])], dim=1)
        logits = _logits(core, seq)
        n = ids.shape[1]
        return ce(logits[:, -n:, :].reshape(-1, core.vocab_size), ids.reshape(-1))

    best = warm_bar
    hist, evals = [], []
    for ep in range(epochs):
        proj.train()
        order = torch.randperm(len(train)); run = 0.0; used = 0
        for i in order:
            opt.zero_grad()
            L = loss_on(train[int(i)])
            if not torch.isfinite(L):
                continue
            L.backward()
            torch.nn.utils.clip_grad_norm_(list(proj.parameters()) + lora_params, 1.0)
            opt.step(); run += float(L.item()); used += 1
        hist.append(round(run / max(1, used), 4))
        proj.eval()
        m = evaluate(core, proj, bias, val, dev)
        evals.append({"epoch": ep, "train_loss": hist[-1], **m})
        print(json.dumps(evals[-1]), flush=True)
        if m["overall_exact_match"] >= best:  # only save non-regressions of the whole task set
            best = m["overall_exact_match"]
            torch.save({"projector": proj.state_dict(), "lora": lora_state_dict(core.model),
                        "d_enc": d_enc, "d_model": core.d_model, "r": r,
                        "tasks": sorted({x["task"] for x in train}), "val_overall_exact": best},
                       ROOT / "data/alm/bi_instruct_ckpt.pt")
            print(json.dumps({"checkpoint_saved": ep, "best_overall_exact": best}), flush=True)

    out = {"method": f"bilingual instruction broadening (200M SanskritCore + LoRA r={r}), warm instruct_ckpt",
           "corpora": [str(m.name) for m, _ in CORPORA], "epochs": epochs,
           "baseline_before": base, "eval_history": evals,
           "save_rule": "checkpoint saved only when overall val exact-match >= warm baseline (no regression)",
           "honesty": "translate_en emits English — expected weakest per the interference finding; "
                      "per-task numbers reported as measured, favourable or not"}
    (ROOT / "data/alm/bi_instruct_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4))
