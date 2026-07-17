"""TDD Phase 0: the vertical slice learns — projector trains the frozen core to emit target text.

The decisive de-risk: gradients flow audio-features → SphotaProjector → (frozen) prabhasa core →
byte cross-entropy, and a small overfit drives loss down and makes the core greedily emit the
target Sanskrit/English byte string. Proves the projected-audio → core bridge is trainable.
"""
from pathlib import Path

import pytest
import torch

from pranava.alm.core_adapter import CHECKPOINTS, SanskritCore, alm_runnable
from pranava.alm.projector import SphotaProjector

pytestmark = pytest.mark.skipif(
    not alm_runnable(),
    reason="prabhasa m2 checkpoint or CUDA not present — run in prabhasa/nemo-gb10",
)

D_ENC = 512  # stand-in encoder dim for the smoke (real Parakeet dim wired in Phase 1)


def test_projector_shapes():
    proj = SphotaProjector(d_enc=D_ENC, d_model=768, downsample=4)
    frames = torch.randn(2, 64, D_ENC)
    out = proj(frames)
    assert out.shape[0] == 2 and out.shape[2] == 768
    assert out.shape[1] == proj.out_len(64)


@pytest.mark.slow
def test_overfit_drives_loss_down_and_emits_target():
    core = SanskritCore(arm="m2", device="cuda").load()
    dev = core.torch_device
    proj = SphotaProjector(d_enc=D_ENC, d_model=core.d_model, downsample=4).to(dev)

    torch.manual_seed(0)
    # 3 "clips": fixed random audio-like features → distinct target byte strings
    targets = [b"ramah", b"vanam", b"agnih"]
    clips = [torch.randn(1, 40, D_ENC, device=dev) for _ in targets]
    tgt_ids = [torch.tensor([[b for b in t]], dtype=torch.long, device=dev) for t in targets]

    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    bias = core.structural_bias
    ce = torch.nn.CrossEntropyLoss()

    def step_loss():
        total = 0.0
        for frames, ids in zip(clips, tgt_ids):
            audio_tok = proj(frames, structural_bias=bias)  # (1, Ta', d)
            tgt_emb = core.embed_tokens(ids[:, :-1])  # teacher-forced prefix (already has bias)
            seq = torch.cat([audio_tok, tgt_emb], dim=1)
            # forward with grad through the frozen core (params frozen, activations differentiable)
            x = seq
            for block in core.model.blocks:
                x = block(x)
            logits = core.model.head(core.model.norm_f(x))
            n_pred = ids.shape[1]  # predict all target bytes from the last n_pred positions
            pred = logits[:, -n_pred:, :].reshape(-1, core.vocab_size)
            total = total + ce(pred, ids.reshape(-1))
        return total / len(clips)

    # freeze core
    for p in core.model.parameters():
        p.requires_grad_(False)

    loss0 = float(step_loss().item())
    for _ in range(150):
        opt.zero_grad()
        loss = step_loss()
        loss.backward()
        opt.step()
    loss1 = float(step_loss().item())

    # Trainability invariant: the projector drives the frozen core to near-perfect overfit.
    assert loss1 < 0.05, f"overfit did not converge: {loss0:.3f} -> {loss1:.3f}"
    assert loss1 < 0.2 * loss0
    # With loss this low the core should greedily emit at least one target exactly.
    with torch.no_grad():
        emits = [bytes(core.greedy_from_embeds(proj(c, structural_bias=bias), max_new=len(t)))
                 for c, t in zip(clips, targets)]
    assert any(e == t for e, t in zip(emits, targets)), f"no exact emit: {emits} vs {targets}"
