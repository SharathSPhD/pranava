"""P3-S-U: Sphoṭa-workspace steering uptake measurement.

Pre-registered experiment to test whether injecting a concept direction into the sphoṭa workspace
band produces readback-verified uptake (rank of concept first-byte in top-5 at t+1..t+3) beyond
matched random-direction control.

Method:
  - ≥30 val clips from speech_corpus_indic_xl (fallback speech_corpus_indic)
  - Concept = gold kāraka filler's byte sequence (target word, e.g. the kriyā)
  - Inject α·‖h‖·direction at band layers during greedy decode, α ∈ {0.05, 0.1, 0.2, 0.4}
  - Controls: (a) random unit direction at same α·‖h‖ norm; (b) α=0 (baseline)
  - Readback: rank of concept first-byte in logits at t+1..t+3
  - Measure entropy before/after; per-clip records + aggregate uptake rates
  - Mala classification per prabodha rules: loaded/amplified/persistent/within-budget

Pre-registered hypotheses:
  - H-SU1 (loaded): concept rank enters top-5 at ≥2× the random-control rate at some α
  - H-SU2 (behavioral): steered output contains concept word at ≥2× control rate
  - H-SU3 (svātantrya budget): |entropy delta| ≤ 0.5 nats at the α that satisfies H-SU1

Output: data/alm/p3su_results.json with per-clip records and aggregate metrics.

Falsifiable: if uptake ≈ control at every α within budget, steering does not work on this model.

Run in container: scripts/alm/in_container.sh python /work/pranava/scripts/alm/p3_steering_uptake.py
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.data import CORPUS_DIR, load_manifest, text_to_bytes
from pranava.alm.projector import SphotaProjector
from pranava.sphota_lens.steer import concept_direction, greedy_steered
from pranava.sphota_lens.lens import fit_sphota_lens

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
FEAT_DIR = CORPUS_DIR / "feats"
OUT = ROOT / "data/alm"


@dataclass(frozen=True, slots=True)
class SteeringRecord:
    """Per-clip steering uptake record."""
    clip_id: str
    concept_word: str
    alpha_0p05_uptake_rate: float  # % of steps t+1..t+3 where rank(concept_byte) <= 5
    alpha_0p05_entropy_delta: float
    alpha_0p05_entropy_before: float
    alpha_0p05_entropy_after: float
    alpha_0p1_uptake_rate: float
    alpha_0p1_entropy_delta: float
    alpha_0p1_entropy_before: float
    alpha_0p1_entropy_after: float
    alpha_0p2_uptake_rate: float
    alpha_0p2_entropy_delta: float
    alpha_0p2_entropy_before: float
    alpha_0p2_entropy_after: float
    alpha_0p4_uptake_rate: float
    alpha_0p4_entropy_delta: float
    alpha_0p4_entropy_before: float
    alpha_0p4_entropy_after: float
    random_uptake_rate: float  # control: random direction at best α
    baseline_uptake_rate: float  # control: α=0
    alpha_0p05_behavioral: bool  # concept word in output (H-SU2)
    alpha_0p1_behavioral: bool
    alpha_0p2_behavioral: bool
    alpha_0p4_behavioral: bool
    random_behavioral: bool
    baseline_behavioral: bool


def _rank_in_logits(logits: torch.Tensor, target_id: int) -> int:
    """Rank of target_id in descending logits (0-indexed; 0 = top-1)."""
    sorted_idx = torch.argsort(logits, descending=True)
    rank = (sorted_idx == target_id).nonzero(as_tuple=True)[0]
    return int(rank.item()) if len(rank) > 0 else 256


def _entropy(logits: torch.Tensor) -> float:
    """Shannon entropy of softmax(logits) in nats."""
    p = torch.softmax(logits, dim=-1)
    return float(-(p * torch.log(p + 1e-10)).sum().item())


@torch.no_grad()
def greedy_steered_with_logits(core, prefix_embeds: torch.Tensor, band: tuple[int, int],
                                direction: torch.Tensor | None, alpha: float,
                                max_new: int = 24) -> tuple[list[int], list[torch.Tensor]]:
    """Greedy decode + per-step logits. Returns (tokens, logits_per_step)."""
    b1, b2 = band
    x = prefix_embeds.to(core.torch_device)
    out: list[int] = []
    all_logits: list[torch.Tensor] = []
    for _ in range(max_new):
        h = x
        for li, block in enumerate(core.model.blocks):
            h = block(h)
            layer_idx = li + 1
            if direction is not None and alpha != 0.0 and b1 <= layer_idx < b2:
                scale = alpha * h[:, -1:, :].norm(dim=-1, keepdim=True)
                h = h.clone()
                h[:, -1:, :] = h[:, -1:, :] + scale * direction
        logits = core.model.head(core.model.norm_f(h))[0, -1]
        all_logits.append(logits.detach().clone())
        nxt = int(torch.argmax(logits).item())
        out.append(nxt)
        nxt_emb = core.model.embed(
            torch.tensor([[nxt]], device=core.torch_device, dtype=torch.long)
        )
        x = torch.cat([x, nxt_emb], dim=1)
    return out, all_logits


def main(n_clips: int = 40, alphas: list[float] = None, seed: int = 42) -> int:
    if alphas is None:
        alphas = [0.05, 0.1, 0.2, 0.4]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias

    # Load checkpoint (xl_ckpt.pt if it exists, else instruct_ckpt.pt)
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

    # Fit the lens to locate the workspace band
    print("Fitting sphoṭa-lens to locate workspace band...", flush=True)
    exs = [e for e in load_manifest("val") if e.kriya and (FEAT_DIR / f"{e.id}.npy").exists()][:n_clips]
    lens_samples = []
    with torch.no_grad():
        for ex in exs:
            feats = torch.from_numpy(np.load(FEAT_DIR / f"{ex.id}.npy").astype(np.float32)
                                      ).unsqueeze(0).to(dev)
            audio_tok = proj(feats, structural_bias=bias)
            ids = torch.tensor([text_to_bytes(ex.text)], dtype=torch.long, device=dev)
            emb = torch.cat([audio_tok, core.embed_tokens(ids)], dim=1)
            lens_samples.append((emb, int(audio_tok.shape[1])))

    lens_report = fit_sphota_lens(core, lens_samples)
    band = lens_report.bands
    print(f"Workspace band: {band}", flush=True)

    # Steering uptake measurement
    print(f"Measuring steering uptake over {len(exs)} clips...", flush=True)
    records: list[SteeringRecord] = []

    for clip_idx, ex in enumerate(exs):
        if clip_idx % 10 == 0:
            print(f"  Processing clip {clip_idx}/{len(exs)}...", flush=True)

        feats = torch.from_numpy(np.load(FEAT_DIR / f"{ex.id}.npy").astype(np.float32)
                                  ).unsqueeze(0).to(dev)
        audio_tok = proj(feats, structural_bias=bias)
        prefix_embeds = audio_tok  # audio tokens only

        # Concept = kriyā (verb) bytes
        concept_word = ex.kriya
        concept_bytes = list(concept_word.encode("utf-8"))
        if not concept_bytes:
            continue
        concept_first_byte = concept_bytes[0]

        # Baseline (no steering, α=0)
        _, baseline_logits = greedy_steered_with_logits(core, prefix_embeds, band, None, 0.0)

        # Random control direction (same seed, deterministic; CPU generator → move to device)
        gen = torch.Generator().manual_seed(seed + clip_idx)
        random_dir = torch.randn(core.d_model, generator=gen)
        random_dir = (random_dir / (random_dir.norm() + 1e-8)).to(dev).detach()

        # Concept direction
        concept_dir = concept_direction(core, concept_bytes)

        uptakes = {}
        entropies = {}
        behavioral = {}  # whether concept word appears in output
        for alpha in alphas + [None]:  # None for random control
            if alpha is None:
                direction = random_dir
                label = "random"
            else:
                direction = concept_dir
                label = f"alpha_{alpha}"

            tokens, logits = greedy_steered_with_logits(core, prefix_embeds, band, direction, alpha or 0.0)

            # Uptake: rank of concept_first_byte at t+1..t+3
            ranks = [_rank_in_logits(logits[i], concept_first_byte) for i in range(min(3, len(logits)))]
            uptake_rate = sum(1 for r in ranks if r < 5) / len(ranks) if ranks else 0.0

            # Entropy
            entropies_before = [_entropy(baseline_logits[i]) for i in range(min(3, len(baseline_logits)))]
            entropies_after = [_entropy(logits[i]) for i in range(min(3, len(logits)))]
            entropy_before = np.mean(entropies_before) if entropies_before else 0.0
            entropy_after = np.mean(entropies_after) if entropies_after else 0.0
            entropy_delta = entropy_after - entropy_before

            # Behavioral: does the concept word appear in the output?
            output_text = bytes(b for b in tokens if 9 <= b < 127).decode("latin-1", "ignore").strip()
            has_concept = concept_word.lower() in output_text.lower()

            uptakes[label] = uptake_rate
            entropies[label] = {
                "before": entropy_before,
                "after": entropy_after,
                "delta": entropy_delta,
            }
            behavioral[label] = has_concept

        # Build record
        rec = SteeringRecord(
            clip_id=ex.id,
            concept_word=concept_word,
            alpha_0p05_uptake_rate=uptakes.get("alpha_0.05", 0.0),
            alpha_0p05_entropy_delta=entropies.get("alpha_0.05", {}).get("delta", 0.0),
            alpha_0p05_entropy_before=entropies.get("alpha_0.05", {}).get("before", 0.0),
            alpha_0p05_entropy_after=entropies.get("alpha_0.05", {}).get("after", 0.0),
            alpha_0p1_uptake_rate=uptakes.get("alpha_0.1", 0.0),
            alpha_0p1_entropy_delta=entropies.get("alpha_0.1", {}).get("delta", 0.0),
            alpha_0p1_entropy_before=entropies.get("alpha_0.1", {}).get("before", 0.0),
            alpha_0p1_entropy_after=entropies.get("alpha_0.1", {}).get("after", 0.0),
            alpha_0p2_uptake_rate=uptakes.get("alpha_0.2", 0.0),
            alpha_0p2_entropy_delta=entropies.get("alpha_0.2", {}).get("delta", 0.0),
            alpha_0p2_entropy_before=entropies.get("alpha_0.2", {}).get("before", 0.0),
            alpha_0p2_entropy_after=entropies.get("alpha_0.2", {}).get("after", 0.0),
            alpha_0p4_uptake_rate=uptakes.get("alpha_0.4", 0.0),
            alpha_0p4_entropy_delta=entropies.get("alpha_0.4", {}).get("delta", 0.0),
            alpha_0p4_entropy_before=entropies.get("alpha_0.4", {}).get("before", 0.0),
            alpha_0p4_entropy_after=entropies.get("alpha_0.4", {}).get("after", 0.0),
            random_uptake_rate=uptakes.get("random", 0.0),
            baseline_uptake_rate=uptakes.get("alpha_0", 0.0),
            alpha_0p05_behavioral=behavioral.get("alpha_0.05", False),
            alpha_0p1_behavioral=behavioral.get("alpha_0.1", False),
            alpha_0p2_behavioral=behavioral.get("alpha_0.2", False),
            alpha_0p4_behavioral=behavioral.get("alpha_0.4", False),
            random_behavioral=behavioral.get("random", False),
            baseline_behavioral=behavioral.get("alpha_0", False),
        )
        records.append(rec)

    # Aggregate metrics
    if records:
        uptakes_by_alpha = {
            "alpha_0.05": np.mean([r.alpha_0p05_uptake_rate for r in records]),
            "alpha_0.1": np.mean([r.alpha_0p1_uptake_rate for r in records]),
            "alpha_0.2": np.mean([r.alpha_0p2_uptake_rate for r in records]),
            "alpha_0.4": np.mean([r.alpha_0p4_uptake_rate for r in records]),
            "random_control": np.mean([r.random_uptake_rate for r in records]),
            "baseline": np.mean([r.baseline_uptake_rate for r in records]),
        }
        behavioral_by_alpha = {
            "alpha_0.05": np.mean([1 if r.alpha_0p05_behavioral else 0 for r in records]),
            "alpha_0.1": np.mean([1 if r.alpha_0p1_behavioral else 0 for r in records]),
            "alpha_0.2": np.mean([1 if r.alpha_0p2_behavioral else 0 for r in records]),
            "alpha_0.4": np.mean([1 if r.alpha_0p4_behavioral else 0 for r in records]),
            "random_control": np.mean([1 if r.random_behavioral else 0 for r in records]),
            "baseline": np.mean([1 if r.baseline_behavioral else 0 for r in records]),
        }
        entropy_deltas_by_alpha = {
            "alpha_0.05": np.mean([r.alpha_0p05_entropy_delta for r in records]),
            "alpha_0.1": np.mean([r.alpha_0p1_entropy_delta for r in records]),
            "alpha_0.2": np.mean([r.alpha_0p2_entropy_delta for r in records]),
            "alpha_0.4": np.mean([r.alpha_0p4_entropy_delta for r in records]),
        }

        # H-SU1: concept rank top-5 at ≥2× control rate
        best_alpha = max(uptakes_by_alpha.items(), key=lambda x: x[1] if "alpha" in x[0] else 0)
        random_rate = uptakes_by_alpha["random_control"]
        h_su1_satisfied = best_alpha[1] >= 2 * random_rate if random_rate > 0 else False

        # H-SU2: concept word in output at ≥2× control rate
        best_behavioral_alpha = max(behavioral_by_alpha.items(), key=lambda x: x[1] if "alpha" in x[0] else 0)
        random_behavioral_rate = behavioral_by_alpha["random_control"]
        h_su2_satisfied = best_behavioral_alpha[1] >= 2 * random_behavioral_rate if random_behavioral_rate > 0 else False

        # H-SU3: entropy delta within budget
        best_alpha_entropy = entropy_deltas_by_alpha.get(best_alpha[0], 0.0)
        h_su3_satisfied = abs(best_alpha_entropy) <= 0.5

        result = {
            "checkpoint_used": ckpt_used,
            "n_clips": len(records),
            "workspace_band": list(band),
            "lens_band_contrast": round(lens_report.band_contrast, 4),
            "alphas_tested": alphas,
            "uptake_rates_by_alpha": {k: round(v, 4) for k, v in uptakes_by_alpha.items()},
            "behavioral_rates_by_alpha": {k: round(v, 4) for k, v in behavioral_by_alpha.items()},
            "entropy_deltas_by_alpha": {k: round(v, 4) for k, v in entropy_deltas_by_alpha.items()},
            "hypothesis_H_SU1": {
                "claim": "concept rank top-5 at >=2x random-control rate",
                "best_alpha": best_alpha[0],
                "best_uptake_rate": round(best_alpha[1], 4),
                "random_control_rate": round(random_rate, 4),
                "ratio": round(best_alpha[1] / random_rate, 2) if random_rate > 0 else None,
                "satisfied": bool(h_su1_satisfied),
            },
            "hypothesis_H_SU2": {
                "claim": "concept word in output at >=2x random-control rate",
                "best_alpha": best_behavioral_alpha[0],
                "best_behavioral_rate": round(best_behavioral_alpha[1], 4),
                "random_control_rate": round(random_behavioral_rate, 4),
                "ratio": round(best_behavioral_alpha[1] / random_behavioral_rate, 2) if random_behavioral_rate > 0 else None,
                "satisfied": bool(h_su2_satisfied),
            },
            "hypothesis_H_SU3": {
                "claim": "entropy delta within svātantrya budget (<=0.5 nats)",
                "best_alpha": best_alpha[0],
                "entropy_delta": round(best_alpha_entropy, 4),
                "satisfied": bool(h_su3_satisfied),
            },
            "per_clip_records": [asdict(r) for r in records],
        }
    else:
        result = {
            "checkpoint_used": ckpt_used,
            "error": "no valid clips processed",
            "per_clip_records": [],
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p3su_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results written to {OUT / 'p3su_results.json'}", flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != "per_clip_records"}, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
