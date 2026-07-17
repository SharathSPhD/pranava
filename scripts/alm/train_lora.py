"""Real end-to-end LoRA fine-tuning: projector + LoRA-adapted core (GB10 for 200M).

Unlike Phase 2 (projector-only, frozen core), this trains the Sphoṭa Projector AND rank-r LoRA
adapters on the core's mixer/attention projections — the core adapts to speech. Base weights frozen;
only projector + LoRA train. Drives CER below the projector-only baseline. Run in prabhasa/nemo-gb10.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.data import CORPUS_DIR, load_manifest, text_to_bytes
from pranava.alm.lora import inject_lora, lora_state_dict
from pranava.alm.projector import SphotaProjector
from pranava.alm.train import _levenshtein

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[1]
FEAT_DIR = CORPUS_DIR / "feats"


def _feat(ex, dev):
    return torch.from_numpy(np.load(FEAT_DIR / f"{ex.id}.npy").astype(np.float32)).unsqueeze(0).to(dev)


@torch.no_grad()
def _cer(core, proj, exs, bias, use_audio):
    dev = core.torch_device
    ed = tot = 0
    for ex in exs:
        tgt = text_to_bytes(ex.text)
        prefix = proj(_feat(ex, dev), structural_bias=bias) if use_audio else bias.view(1, 1, -1)
        out = [b for b in core.greedy_from_embeds(prefix, max_new=len(tgt) + 4) if b != 0][: len(tgt)]
        ed += _levenshtein(out, tgt); tot += max(1, len(tgt))
    return ed / tot


def _logits(core, seq):
    x = seq
    for b in core.model.blocks:
        x = b(x)
    return core.model.head(core.model.norm_f(x))


def main(epochs: int = 10, lr: float = 5e-4, r: int = 8) -> int:
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias
    lora_params = inject_lora(core.model, r=r, alpha=2 * r)
    core.model.to(dev)
    d_enc = int(np.load(FEAT_DIR / f"{load_manifest()[0].id}.npy").shape[-1])
    proj = SphotaProjector(d_enc=d_enc, d_model=core.d_model, downsample=4).to(dev)
    opt = torch.optim.Adam(list(proj.parameters()) + lora_params, lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    train_ex, val_ex = load_manifest("train"), load_manifest("val")

    def loss_on(ex):
        ids = torch.tensor([text_to_bytes(ex.text)], dtype=torch.long, device=dev)
        seq = torch.cat([proj(_feat(ex, dev), structural_bias=bias), core.embed_tokens(ids[:, :-1])], dim=1)
        logits = _logits(core, seq)
        n = ids.shape[1]
        return ce(logits[:, -n:, :].reshape(-1, core.vocab_size), ids.reshape(-1))

    hist = []
    n_lora = sum(p.numel() for p in lora_params)
    for ep in range(epochs):
        proj.train()
        order = torch.randperm(len(train_ex)); run = 0.0
        for i in order:
            ex = train_ex[int(i)]
            opt.zero_grad(); L = loss_on(ex); L.backward(); opt.step(); run += float(L.item())
        hist.append(round(run / len(train_ex), 4))
        print(json.dumps({"epoch": ep, "train_loss": hist[-1]}), flush=True)

    proj.eval()
    cer_a = _cer(core, proj, val_ex, bias, True)
    cer_n = _cer(core, proj, val_ex, bias, False)
    torch.save({"projector": proj.state_dict(), "lora": lora_state_dict(core.model),
                "d_enc": d_enc, "d_model": core.d_model, "r": r}, ROOT / "data/alm/lora_ckpt.pt")
    m = {"method": "projector+LoRA(r=%d) on 200M core" % r, "n_lora_params": n_lora, "epochs": epochs,
         "train_loss_history": hist, "val_cer_audio": round(cer_a, 4),
         "val_cer_noaudio_baseline": round(cer_n, 4),
         "projector_only_baseline_cer": 0.7745, "improves_on_projector_only": bool(cer_a < 0.7745),
         "cer_improvement_over_noaudio": round(cer_n - cer_a, 4)}
    (ROOT / "data/alm/lora_metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
