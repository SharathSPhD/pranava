"""TDD: the SanskritCore adapter feeds inputs_embeds through the real prabhasa checkpoint.

Requires the m2 checkpoint on disk. The decisive test is `test_embeds_path_matches_token_path`:
feeding embed(tokens) through forward_embeds must equal the core's own forward(tokens) — proving
our injection path is faithful to the trained model.
"""
from pathlib import Path

import pytest
import torch

from pranava.alm.core_adapter import CHECKPOINTS, SanskritCore, alm_runnable

pytestmark = pytest.mark.skipif(
    not alm_runnable(),
    reason="prabhasa m2 checkpoint or CUDA (mamba_ssm) not present — run in prabhasa/nemo-gb10",
)


@pytest.fixture(scope="module")
def core():
    return SanskritCore(arm="m2", device="cuda").load()


def test_dims(core):
    assert core.d_model == 768
    assert core.vocab_size == 256


def test_embed_tokens_shape(core):
    ids = torch.tensor([[104, 101, 108, 108, 111]], dtype=torch.long)  # "hello" bytes
    emb = core.embed_tokens(ids)
    assert emb.shape == (1, 5, 768)


def test_forward_embeds_shape(core):
    x = torch.randn(1, 7, 768)
    logits = core.forward_embeds(x)
    assert logits.shape == (1, 7, 256)


def test_embeds_path_matches_token_path(core):
    """forward_embeds(embed(tokens)) must equal the core's own forward(tokens)."""
    ids = torch.tensor([[115, 97, 98, 100, 97]], dtype=torch.long, device=core.torch_device)
    zeros = torch.zeros_like(ids)
    with torch.no_grad():
        ref = core.model(ids, zeros, zeros)  # core's own forward (unstructured → zeros)
        emb = core.embed_tokens(ids)
        got = core.forward_embeds(emb)
    # Faithful injection: identical predictions; residual is mamba fused-kernel non-determinism
    # (max |Δlogit| ~2e-3 vs logit scale ~3.4, i.e. ~0.06%).
    assert (ref.argmax(-1) == got.argmax(-1)).all()
    assert (ref - got).abs().max().item() < 0.02


def test_features_embeds_is_workspace(core):
    x = torch.randn(1, 4, 768)
    h = core.features_embeds(x)
    assert h.shape == (1, 4, 768)  # pre-head hidden states = sphoṭa workspace


def test_greedy_decode_runs(core):
    ids = torch.tensor([[115, 97]], dtype=torch.long)
    emb = core.embed_tokens(ids)
    out = core.greedy_from_embeds(emb, max_new=8)
    assert len(out) == 8
    assert all(0 <= b < 256 for b in out)
