"""P3-F-2: Fusion-v2 — causal cross-modal integration profile.

Pre-registered experiment to measure WHERE audio information becomes causally necessary for the
text-side meaning — i.e. the sphoṭa locus measured causally, not by CKA similarity (v1's weakness).

Method:
  - For each layer ℓ: run the model with audio-position hidden states at ℓ replaced by their
    batch-mean (information cut at ℓ); measure kriyā-decodability from text positions at the final
    layer (reuse emergence.py's linear probe machinery)
  - Integration(ℓ) = decodability(intact) − decodability(cut at ℓ)
  - Permutation chance from label shuffles (100 permutations)
  - Report: curve, peak layer, comparison to correlational peak from emergence.py

Pre-registered hypotheses:
  - H-F2a: Integration(ℓ) has an interior peak (not layer 0/L) — a genuine fusion locus
  - H-F2b: the causal peak lies within ±2 layers of the correlational sphoṭa layer (13 on the old
    model; re-measured on the eval model) — corroboration; if not, the discrepancy is reported and
    the causal number wins

Output: data/alm/p3f2_results.json with causal importance curve, peak layer, and comparison.

Run in container: scripts/alm/in_container.sh python /work/pranava/scripts/alm/p3_fusion_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.data import CORPUS_DIR, load_manifest, text_to_bytes
from pranava.alm.projector import SphotaProjector
from pranava.sphota_lens.emergence import meaning_emergence

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
FEAT_DIR = CORPUS_DIR / "feats"
OUT = ROOT / "data/alm"


def _grouped_cv_acc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> float:
    """Template-grouped CV accuracy of a linear probe (no lexical leakage)."""
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=min(4, n_groups))
    correct = total = 0
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(sc.transform(X[tr]), y[tr])
        correct += int((clf.predict(sc.transform(X[te])) == y[te]).sum())
        total += len(te)
    return correct / total if total else float("nan")


@torch.no_grad()
def decodability_with_fusion(core, samples: list[tuple[torch.Tensor, int, str, str]]) -> float:
    """Measure kriyā-decodability from text positions at final layer.

    Samples: (fused_embeds (1,T,d), n_audio_tokens, meaning_label, cv_group).
    Returns accuracy of grouped-CV linear probe on text-position reps.
    """
    labels = np.array([s[2] for s in samples])
    groups = np.array([s[3] for s in samples])
    classes, y = np.unique(labels, return_inverse=True)

    # Extract text-position representations at final layer
    text_reps = []
    for emb, n_aud, _, _ in samples:
        # Forward: features_per_layer returns [input, block_0, ..., block_L, norm_f(block_L)]
        layers = core.features_per_layer(emb)
        final_layer = layers[-1]  # (1, T, d)
        # Text positions: [n_aud, T)
        T = emb.shape[1]
        if T > n_aud:
            text_pos_rep = final_layer[0, n_aud:].mean(dim=0).detach().float().cpu().numpy()
        else:
            # No text positions; skip
            text_pos_rep = np.zeros(core.d_model)
        text_reps.append(text_pos_rep)

    acts = np.stack(text_reps)  # (N, d)
    if acts.shape[0] < 2 or len(np.unique(y)) < 2:
        return float("nan")

    acc = _grouped_cv_acc(acts, y, groups)
    return acc


@torch.no_grad()
def _text_rep_for_ablation(core, emb: torch.Tensor, n_aud: int, li: int) -> np.ndarray:
    """Final-layer text-position rep with audio ablated (mean-collapsed) at layer li. Label-independent."""
    x = emb.to(core.torch_device)
    if li == 0:
        x = x.clone()
        x[:, :n_aud] = x[:, :n_aud].mean(dim=1, keepdim=True)
    else:
        for bi in range(li - 1):
            x = core.model.blocks[bi](x)
        x = x.clone()
        x[:, :n_aud] = x[:, :n_aud].mean(dim=1, keepdim=True)
        for bi in range(li - 1, len(core.model.blocks)):
            x = core.model.blocks[bi](x)
    x = core.model.norm_f(x)
    T = emb.shape[1]
    if T > n_aud:
        return x[0, n_aud:].mean(dim=0).detach().float().cpu().numpy()
    return np.zeros(core.d_model)


@torch.no_grad()
def collect_ablation_acts(core, samples: list[tuple[torch.Tensor, int, str, str]]):
    """Run ALL the expensive forward passes ONCE (label-independent).

    Returns (acts_intact (N,d), acts_ablated_by_layer [n_layers × (N,d)], groups (N,)). Permutation
    testing then re-scores these cached activations with shuffled labels — no re-forwarding (the old
    code re-ran every ablation forward 101×; the hidden states never depend on the labels)."""
    groups = np.array([s[3] for s in samples])
    intact = []
    for emb, n_aud, _, _ in samples:
        final = core.features_per_layer(emb)[-1]
        T = emb.shape[1]
        intact.append(final[0, n_aud:].mean(dim=0).detach().float().cpu().numpy()
                      if T > n_aud else np.zeros(core.d_model))
    acts_intact = np.stack(intact)
    n_layers = len(core.features_per_layer(samples[0][0]))
    acts_ablated = []
    for li in range(n_layers):
        acts_ablated.append(np.stack([_text_rep_for_ablation(core, emb, n_aud, li)
                                      for emb, n_aud, _, _ in samples]))
    return acts_intact, acts_ablated, groups


def score_integration(acts_intact, acts_ablated, groups, y) -> dict:
    """Integration_by_layer = decodability(intact) − decodability(cut at ℓ), for given labels y (cheap)."""
    if len(np.unique(y)) < 2:
        return {"baseline_decodability": float("nan"),
                "integration_by_layer": [0.0] * len(acts_ablated), "peak_layer": 0}
    baseline_acc = _grouped_cv_acc(acts_intact, y, groups)
    causal = []
    for acts in acts_ablated:
        acc_ablated = _grouped_cv_acc(acts, y, groups)
        importance = baseline_acc - (acc_ablated if not np.isnan(acc_ablated) else baseline_acc)
        causal.append(max(0.0, importance))
    peak = int(np.argmax(causal)) if causal else 0
    return {"baseline_decodability": baseline_acc, "integration_by_layer": causal, "peak_layer": peak}


@torch.no_grad()
def causal_integration(core, samples: list[tuple[torch.Tensor, int, str, str]]) -> dict:
    """Convenience wrapper: collect (once) + score with the true labels."""
    labels = np.array([s[2] for s in samples])
    _, y = np.unique(labels, return_inverse=True)
    acts_intact, acts_ablated, groups = collect_ablation_acts(core, samples)
    return {
        **score_integration(acts_intact, acts_ablated, groups, y),
    }


def main(n_clips: int = 100, n_permutations: int = 100, min_per_class: int = 6) -> int:
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias

    # Load checkpoint
    ckpt_paths = [ROOT / "data/alm/xl_ckpt.pt", ROOT / "data/alm/instruct_ckpt.pt"]
    blob = None
    ckpt_used = None
    for ckpt_path in ckpt_paths:
        if ckpt_path.exists():
            blob = torch.load(ckpt_path, map_location=dev, weights_only=True)
            ckpt_used = str(ckpt_path)
            break
    if blob is None:
        print(f"ERROR: no checkpoint found at {ckpt_paths}", flush=True)
        return 1

    proj = SphotaProjector(d_enc=blob.get("d_enc", 768), d_model=blob.get("d_model", 2048),
                           downsample=4).to(dev).eval()
    if "projector" in blob:
        proj.load_state_dict(blob["projector"])
    elif "state_dict" in blob:
        proj.load_state_dict(blob["state_dict"])
    print(f"Loaded checkpoint: {ckpt_used}", flush=True)

    # Prepare samples: same as in emergence.py (kriyas with sufficient class support)
    print("Preparing samples with kriyā labels...", flush=True)
    exs = [e for e in load_manifest() if e.kriya and (FEAT_DIR / f"{e.id}.npy").exists()]
    from collections import Counter
    ct = Counter(e.kriya for e in exs)
    exs = [e for e in exs if ct[e.kriya] >= min_per_class][:n_clips]
    print(f"Using {len(exs)} clips with {len(set(e.kriya for e in exs))} unique kriyas", flush=True)

    samples = []
    with torch.no_grad():
        for ex in exs:
            feats = torch.from_numpy(np.load(FEAT_DIR / f"{ex.id}.npy").astype(np.float32)
                                      ).unsqueeze(0).to(dev)
            audio_tok = proj(feats, structural_bias=bias)
            ids = torch.tensor([text_to_bytes(ex.text)], dtype=torch.long, device=dev)
            emb = torch.cat([audio_tok, core.embed_tokens(ids)], dim=1)
            samples.append((emb, int(audio_tok.shape[1]), ex.kriya, ex.template))

    print("Collecting ablation activations ONCE (GPU forwards; label-independent)...", flush=True)
    acts_intact, acts_ablated, groups = collect_ablation_acts(core, samples)
    labels = np.array([s[2] for s in samples])
    _, y = np.unique(labels, return_inverse=True)
    fusion_v2 = score_integration(acts_intact, acts_ablated, groups, y)

    print("Computing permutation chance level (100 shuffles; cheap re-scoring of cached acts)...", flush=True)
    rng = np.random.default_rng(0)
    chance_curves = []
    for perm_idx in range(n_permutations):
        if perm_idx % 20 == 0:
            print(f"  Permutation {perm_idx}/{n_permutations}...", flush=True)
        y_shuf = rng.permutation(y)
        chance_curves.append(score_integration(acts_intact, acts_ablated, groups, y_shuf)["integration_by_layer"])

    # Compute chance statistics
    chance_curves_arr = np.array(chance_curves)  # (n_perms, n_layers)
    chance_mean = np.mean(chance_curves_arr, axis=0).tolist()
    chance_std = np.std(chance_curves_arr, axis=0).tolist()
    chance_peak_mean = float(np.mean([np.max(c) for c in chance_curves]))

    # Also compute the correlational baseline from emergence.py
    print("Computing correlational baseline (meaning_emergence)...", flush=True)
    emergence_report = meaning_emergence(core, samples)
    correlational_peak = emergence_report.peak_layer

    # H-F2a: interior peak
    causal_peak = fusion_v2["peak_layer"]
    causal_curve = fusion_v2["integration_by_layer"]
    n_layers = len(causal_curve)
    h_f2a_satisfied = 0 < causal_peak < n_layers - 1

    # H-F2b: causal peak within ±2 of correlational peak
    peak_distance = abs(causal_peak - correlational_peak)
    h_f2b_satisfied = peak_distance <= 2

    result = {
        "checkpoint_used": ckpt_used,
        "n_clips": len(exs),
        "n_layers": n_layers,
        "correlational_peak_layer": correlational_peak,
        "correlational_decodability_curve": [round(x, 4) for x in emergence_report.decodability_by_layer],
        "causal_peak_layer": causal_peak,
        "causal_integration_curve": [round(x, 4) for x in causal_curve],
        "baseline_decodability": round(fusion_v2["baseline_decodability"], 4),
        "peak_distance": peak_distance,
        "permutation_chance": {
            "n_permutations": n_permutations,
            "mean_curve": [round(x, 4) for x in chance_mean],
            "std_curve": [round(x, 4) for x in chance_std],
            "mean_peak_value": round(chance_peak_mean, 4),
        },
        "hypothesis_H_F2a": {
            "claim": "Integration(ℓ) has interior peak (not layer 0 or L)",
            "causal_peak": causal_peak,
            "n_layers": n_layers,
            "is_interior": bool(h_f2a_satisfied),
            "satisfied": bool(h_f2a_satisfied),
        },
        "hypothesis_H_F2b": {
            "claim": "causal peak within ±2 layers of correlational peak",
            "causal_peak": causal_peak,
            "correlational_peak": correlational_peak,
            "distance": peak_distance,
            "satisfied": bool(h_f2b_satisfied),
        },
        "reading": {
            "integration_curve": "per-layer importance for kriyā decodability from text positions; "
                                 "peak = where audio becomes causally necessary for text meaning",
            "causal_vs_correlational": f"causal peak at layer {causal_peak}, correlational at {correlational_peak}, "
                                       f"distance {peak_distance} layers",
            "agreement": "corroborated" if h_f2b_satisfied else "discrepancy; causal wins",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p3f2_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results written to {OUT / 'p3f2_results.json'}", flush=True)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("causal_integration_curve", "correlational_decodability_curve",
                                   "permutation_chance")}, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
