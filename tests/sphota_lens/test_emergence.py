"""TDD: the meaning-emergence instrument recovers a planted sphoṭa layer (host-runnable).

We build a mock core whose audio-position reps encode the label linearly ONLY at a chosen layer
(and are noise elsewhere). A correct instrument must peak its decodability there.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")

from pranava.sphota_lens.emergence import meaning_emergence  # noqa: E402

D = 16
PLANT = 5
N_LAYERS = 10


class _MockCore:
    def __init__(self):
        self.torch_device = torch.device("cpu")

    def features_per_layer(self, emb):
        # emb carries (label, group, seed) in the first 3 dims of position 0 (our encoding)
        label = int(emb[0, 0, 0].item())
        g = torch.Generator().manual_seed(1000 + int(emb[0, 0, 2].item()))
        T = emb.shape[1]
        layers = []
        for li in range(N_LAYERS + 1):
            h = torch.randn(1, T, D, generator=g) * 0.3
            if li == PLANT:
                # plant a label-linear signal in the audio positions (first 4)
                onehot = torch.zeros(D)
                onehot[label % D] = 3.0
                h[0, :4] += onehot
            layers.append(h)
        return layers

    def features_final_ablated(self, emb, ablate_layer, positions):
        # final layer carries a (weaker) copy of the planted signal unless the plant layer is ablated
        label = int(emb[0, 0, 0].item())
        g = torch.Generator().manual_seed(2000 + int(emb[0, 0, 2].item()))
        T = emb.shape[1]
        h = torch.randn(1, T, D, generator=g) * 0.3
        if ablate_layer != PLANT:
            onehot = torch.zeros(D); onehot[label % D] = 2.0
            h[0, :4] += onehot
        return h


def _make_samples(n_per_class=8, n_classes=6):
    samples = []
    for c in range(n_classes):
        for j in range(n_per_class):
            emb = torch.zeros(1, 6, D)
            emb[0, 0, 0] = c            # label
            emb[0, 0, 2] = c * 100 + j  # seed
            samples.append((emb, 4, f"verb{c}", f"tmpl{j % 4}"))
    return samples


def test_lens_recovers_planted_layer():
    rep = meaning_emergence(_MockCore(), _make_samples())
    # decodability must peak at (or within 1 of) the planted layer
    assert abs(rep.peak_layer - PLANT) <= 1, rep.decodability_by_layer
    assert rep.decodability_by_layer[PLANT] > rep.chance + 0.2  # strong signal at the plant


def test_report_fields():
    rep = meaning_emergence(_MockCore(), _make_samples())
    d = rep.to_dict()
    for k in ("decodability_by_layer", "peak_layer", "causal_peak_layer", "sphota_layer",
              "validated", "chance"):
        assert k in d
    assert len(d["decodability_by_layer"]) == N_LAYERS + 1
