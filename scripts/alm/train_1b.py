"""Phase 5 — train the Sphoṭa Projector against the fully-trained 1B Nemotron-H core (RTX 5090).

Same stage-1 recipe as Phase 2 but on the 1B Megatron-Core model: frozen 1B core + frozen Parakeet,
train ONLY the projector. Parakeet features cached once. Eval = greedy CER on val (audio vs
no-audio). Runs in prabhasa/nemo-5090:26.02. Writes data/alm/p5_metrics.json + projector_1b.pt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from pranava.alm.data import CORPUS_DIR, load_manifest, read_wav, text_to_bytes
from pranava.alm.megatron_core import Megatron1BCore
from pranava.alm.projector import SphotaProjector
from pranava.alm.train import _levenshtein

ROOT = Path("/work/pranava")
FEAT_DIR = CORPUS_DIR / "feats"


def extract_features() -> int:
    from pranava.alm.encoder import ParakeetEncoder

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    enc = ParakeetEncoder().load()
    d = 0
    for ex in load_manifest():
        out = FEAT_DIR / f"{ex.id}.npy"
        if out.exists():
            d = np.load(out).shape[-1]
            continue
        wav, sr = read_wav(ex.wav_path)
        fr = enc.encode(wav, sr=sr)[0].to("cpu").numpy().astype(np.float16)
        d = fr.shape[-1]
        np.save(out, fr)
    return d


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


def main(epochs: int = 6, lr: float = 1e-3) -> int:
    d_enc = extract_features()
    core = Megatron1BCore().load()
    for p in core._model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias
    proj = SphotaProjector(d_enc=d_enc, d_model=core.d_model, downsample=4).to(dev)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    train_ex, val_ex = load_manifest("train"), load_manifest("val")

    def loss_on(ex):
        ids = torch.tensor([text_to_bytes(ex.text)], dtype=torch.long, device=dev)
        audio_tok = proj(_feat(ex, dev), structural_bias=bias)
        seq = torch.cat([audio_tok, core.embed_tokens(ids[:, :-1])], dim=1)
        logits = core.logits_from_embeds(seq)
        n = ids.shape[1]
        return ce(logits[:, -n:, :].reshape(-1, core.vocab_size), ids.reshape(-1))

    hist = []
    for ep in range(epochs):
        proj.train()
        order = torch.randperm(len(train_ex))
        run = 0.0
        for i in order:
            ex = train_ex[int(i)]
            opt.zero_grad(); L = loss_on(ex); L.backward(); opt.step()
            run += float(L.item())
        hist.append(round(run / len(train_ex), 4))
        print(json.dumps({"epoch": ep, "train_loss": hist[-1]}), flush=True)

    proj.eval()
    cer_a = _cer(core, proj, val_ex, bias, True)
    cer_n = _cer(core, proj, val_ex, bias, False)
    torch.save({"state_dict": proj.state_dict(), "d_enc": d_enc, "d_model": core.d_model,
                "arm": "m4_1b"}, ROOT / "data/alm/projector_1b.pt")
    metrics = {"core": "m4_1b (1.13B Nemotron-H)", "epochs": epochs, "d_model": core.d_model,
               "n_train": len(train_ex), "n_val": len(val_ex), "train_loss_history": hist,
               "val_cer_audio": round(cer_a, 4), "val_cer_noaudio_baseline": round(cer_n, 4),
               "cer_improvement": round(cer_n - cer_a, 4), "beats_baseline": bool(cer_a < cer_n)}
    (ROOT / "data/alm/p5_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 6))
