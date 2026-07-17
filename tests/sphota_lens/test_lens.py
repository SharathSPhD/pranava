"""TDD Phase 3: the Sphoṭa-Lens logic is correct on a tiny mock core (host-runnable, no GPU).

Verifies the lens produces a well-formed report — a workspace band, per-layer fusion, and an
articulation gradient — over synthetic per-layer activations, exercising the reused prabodha
primitives. The real fit on the trained ALM runs in-container (scripts/alm/p3_sphota_lens.py).
"""
import numpy as np
import pytest
import torch

pytest.importorskip("prabodha.lens.e1_metrics", reason="prabodha not importable")

from pranava.sphota_lens.lens import SphotaLensReport, fit_sphota_lens  # noqa: E402


class _MockModel:
    def __init__(self, d, vocab):
        self.head = torch.nn.Linear(d, vocab, bias=False)
        self.embed = torch.nn.Embedding(vocab, d)


class _MockCore:
    """A stand-in exposing the surface fit_sphota_lens needs, with structured layer activations."""

    def __init__(self, n_layers=9, d=8, vocab=32):
        self.torch_device = torch.device("cpu")
        self.model = _MockModel(d, vocab)
        self._n = n_layers
        self._d = d

    def features_per_layer(self, emb):
        # produce n_layers+1 hidden states; make a middle band mutually coherent (a workspace)
        torch.manual_seed(int(emb.sum().abs().item() * 1000) % 100000)
        T = emb.shape[1]
        base = torch.randn(1, T, self._d)
        layers = []
        shared = torch.randn(1, T, self._d)
        for li in range(self._n + 1):
            if 3 <= li <= 5:  # coherent middle band
                layers.append(shared + 0.05 * torch.randn(1, T, self._d))
            else:
                layers.append(base + torch.randn(1, T, self._d))
        return layers


def _samples(n=12, d=8):
    out = []
    for _ in range(n):
        T = 10
        emb = torch.randn(1, T, d)
        out.append((emb, 4))  # 4 audio tokens, 6 text
    return out


def test_report_is_wellformed():
    core = _MockCore(n_layers=9, d=8, vocab=32)
    rep = fit_sphota_lens(core, _samples(d=8))
    assert isinstance(rep, SphotaLensReport)
    assert rep.n_layers == 10  # 9 blocks + input
    assert len(rep.fusion_by_layer) == 10
    assert len(rep.articulation_by_layer) == 10
    b1, b2 = rep.bands
    assert 0 < b1 < b2 < rep.n_layers


def test_articulation_in_unit_interval():
    core = _MockCore()
    rep = fit_sphota_lens(core, _samples())
    assert all(0.0 <= a <= 1.0 for a in rep.articulation_by_layer)


def test_to_dict_has_workspace_band():
    core = _MockCore()
    d = fit_sphota_lens(core, _samples()).to_dict()
    assert "workspace_band" in d and len(d["workspace_band"]) == 2
    assert "fusion_by_layer" in d and "articulation_by_layer" in d
