"""SFT the Śabda-ALM to follow instructions — one audio clip, six tasks.

Warm-starts from the multilingual projector+LoRA checkpoint (the model already transcribes), then
trains projector + LoRA on the instruction corpus with a **response-only** loss over the sequence

    [audio tokens] ++ [instruction bytes] ++ [response bytes]

so the instruction selects which gold field (transcript / language / kartā / karaṇa / karma / kriyā)
the model reads out of the same audio. Eval reports per-task exact-match accuracy AND an
*instruction-sensitivity* control (accuracy with the correct instruction vs a shuffled one): if the
model truly follows instructions, the correct-instruction accuracy is far higher. Run in
prabhasa/nemo-gb10 on the GB10.

    python scripts/alm/train_instruct.py [epochs]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.instruct import EOS, INSTRUCTIONS, build_prefix
from pranava.alm.lora import inject_lora, lora_state_dict
from pranava.alm.projector import SphotaProjector

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[1]
FEATS = ROOT / "data/alm/speech_corpus_multi/feats"
CORPUS = ROOT / "data/alm/instruct_corpus/manifest.jsonl"


def _load_examples(split: str) -> list[dict]:
    return [r for r in (json.loads(l) for l in CORPUS.open()) if r["split"] == split]


def _feat(wav: str, dev):
    stem = Path(wav).stem
    return torch.from_numpy(np.load(FEATS / f"{stem}.npy").astype(np.float32)).unsqueeze(0).to(dev)


def _response_ids(resp: str, dev):
    # target = response bytes + EOS, so the model learns to signal completion (no rambling)
    return torch.tensor([list(resp.encode("utf-8")) + [EOS]], dtype=torch.long, device=dev)


def _logits(core, seq):
    x = seq
    for b in core.model.blocks:
        x = b(x)
    return core.model.head(core.model.norm_f(x))


@torch.no_grad()
def _decode(core, proj, bias, ex, dev, instruction=None):
    audio = proj(_feat(ex["wav"], dev), structural_bias=bias)
    prefix = build_prefix(core, audio, instruction or ex["instruction"])
    # decode until the model emits EOS (generous cap); then keep printable bytes
    out = core.greedy_from_embeds(prefix, max_new=64, stop_token=EOS)
    return bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()


@torch.no_grad()
def _evaluate(core, proj, bias, val, dev):
    """Per-task exact-match + instruction-sensitivity control (shuffled instruction)."""
    from collections import defaultdict

    hit = defaultdict(int); tot = defaultdict(int); ctrl_hit = 0; ctrl_tot = 0
    other = list(INSTRUCTIONS.values())
    for i, ex in enumerate(val):
        pred = _decode(core, proj, bias, ex, dev)
        tot[ex["task"]] += 1
        hit[ex["task"]] += int(pred == ex["response"])
        # control: a *different* instruction (rotate) — same audio, wrong question
        wrong = other[(other.index(ex["instruction"]) + 1 + (i % (len(other) - 1))) % len(other)]
        if wrong != ex["instruction"]:
            ctrl_pred = _decode(core, proj, bias, ex, dev, instruction=wrong)
            ctrl_hit += int(ctrl_pred == ex["response"]); ctrl_tot += 1
    acc = {t: round(hit[t] / tot[t], 4) for t in sorted(tot)}
    overall = round(sum(hit.values()) / max(1, sum(tot.values())), 4)
    ctrl = round(ctrl_hit / max(1, ctrl_tot), 4)
    return {"per_task_accuracy": acc, "overall_accuracy": overall,
            "control_shuffled_instruction_accuracy": ctrl,
            "instruction_sensitivity_gap": round(overall - ctrl, 4)}


def main(epochs: int = 4, lr: float = 3e-4, r: int = 8) -> int:
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias
    lora_params = inject_lora(core.model, r=r, alpha=2 * r)
    core.model.to(dev)

    d_enc = int(np.load(next(FEATS.glob("*.npy"))).shape[-1])
    proj = SphotaProjector(d_enc=d_enc, d_model=core.d_model, downsample=4).to(dev)

    # warm-start from the multilingual checkpoint (already transcribes en+sa)
    warm = ROOT / "data/alm/lora_ckpt.pt"
    if warm.exists():
        blob = torch.load(warm, map_location=dev, weights_only=True)
        proj.load_state_dict(blob["projector"])
        sd = {n: p for n, p in core.model.named_parameters() if ".A" in n or ".B" in n}
        for n, w in blob.get("lora", {}).items():
            if n in sd:
                sd[n].data.copy_(w.to(dev))
        print(f"warm-started from {warm.name}", flush=True)

    opt = torch.optim.Adam(list(proj.parameters()) + lora_params, lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    train, val = _load_examples("train"), _load_examples("val")

    def loss_on(ex):
        audio = proj(_feat(ex["wav"], dev), structural_bias=bias)
        prefix = build_prefix(core, audio, ex["instruction"])  # [audio ++ instruction]
        ids = _response_ids(ex["response"], dev)
        seq = torch.cat([prefix, core.embed_tokens(ids[:, :-1])], dim=1)
        logits = _logits(core, seq)
        n = ids.shape[1]
        return ce(logits[:, -n:, :].reshape(-1, core.vocab_size), ids.reshape(-1))

    hist = []
    for ep in range(epochs):
        proj.train()
        order = torch.randperm(len(train)); run = 0.0
        for i in order:
            opt.zero_grad(); L = loss_on(train[int(i)]); L.backward(); opt.step(); run += float(L.item())
        hist.append(round(run / len(train), 4))
        print(json.dumps({"epoch": ep, "train_loss": hist[-1]}), flush=True)

    proj.eval()
    metrics = _evaluate(core, proj, bias, val, dev)
    torch.save({"projector": proj.state_dict(), "lora": lora_state_dict(core.model),
                "d_enc": d_enc, "d_model": core.d_model, "r": r},
               ROOT / "data/alm/instruct_ckpt.pt")
    out = {"method": "instruction SFT (projector+LoRA r=%d), warm-start multilingual" % r,
           "epochs": epochs, "n_train": len(train), "n_val": len(val),
           "train_loss_history": hist, **metrics,
           "follows_instructions": bool(metrics["instruction_sensitivity_gap"] > 0.05)}
    (ROOT / "data/alm/instruct_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4))
