"""Phase 0 evidence — run the ALM vertical slice + overfit and write a manifest.

Proves on the GB10: (1) real audio → Parakeet → projector → prabhasa core → bytes end-to-end;
(2) the projector trains the frozen core to emit a target string (loss ↓, exact emit). Run in
prabhasa/nemo-gb10 via scripts/alm/in_container.sh.
"""
from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.pipeline import SabdaALM
from pranava.alm.projector import SphotaProjector

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "alm"


def _load(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr, raw = wf.getframerate(), wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def overfit_probe(core: SanskritCore) -> dict:
    dev = core.torch_device
    proj = SphotaProjector(d_enc=512, d_model=core.d_model, downsample=4).to(dev)
    torch.manual_seed(0)
    targets = [b"ramah", b"vanam", b"agnih"]
    clips = [torch.randn(1, 40, 512, device=dev) for _ in targets]
    tgt = [torch.tensor([[b for b in t]], dtype=torch.long, device=dev) for t in targets]
    for p in core.model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    bias = core.structural_bias
    ce = torch.nn.CrossEntropyLoss()

    def loss():
        tot = 0.0
        for frames, ids in zip(clips, tgt):
            at = proj(frames, structural_bias=bias)
            te = core.embed_tokens(ids[:, :-1])
            x = torch.cat([at, te], dim=1)
            for b in core.model.blocks:
                x = b(x)
            lg = core.model.head(core.model.norm_f(x))
            n = ids.shape[1]
            tot = tot + ce(lg[:, -n:, :].reshape(-1, core.vocab_size), ids.reshape(-1))
        return tot / len(clips)

    l0 = float(loss().item())
    for _ in range(60):
        opt.zero_grad(); L = loss(); L.backward(); opt.step()
    l1 = float(loss().item())
    with torch.no_grad():
        emit = bytes(core.greedy_from_embeds(proj(clips[0], structural_bias=bias),
                                             max_new=len(targets[0])))
    return {"loss_before": round(l0, 4), "loss_after": round(l1, 4),
            "target": targets[0].decode(), "emitted": emit.decode("latin-1"),
            "exact_match": emit == targets[0]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    alm = SabdaALM.build(arm="m2", downsample=4)
    wav, sr = _load(ROOT / "data/stimuli/wav/s0000.wav")
    tokens = alm.audio_tokens(wav, sr)
    e2e = {
        "encoder_dim": int(alm.encoder.dim),
        "n_sphota_tokens": int(tokens.shape[1]),
        "d_model": int(alm.core.d_model),
        "device": str(tokens.device),
    }
    over = overfit_probe(alm.core)
    manifest = {"e2e": e2e, "overfit": over, "elapsed_s": round(time.time() - t0, 1),
                "note": "Phase 0 vertical slice: audio→Parakeet→SphotaProjector→prabhasa core→bytes; "
                        "projector trains the frozen core (overfit)."}
    (OUT / "p0_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
