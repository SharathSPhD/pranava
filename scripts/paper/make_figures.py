#!/usr/bin/env python3
"""
Generate figures for the Śabda-ALM paper using real data from JSON files.
Uses matplotlib only (no seaborn).
Output: docs/figures/*.png
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

def load_json(path):
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)

def load_jsonl(path):
    """Load a JSONL file (one JSON object per line)."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def figure_training_trajectory():
    """Figure 1: Training trajectory - val CER per epoch for 1.13B XL model."""
    print("Generating training trajectory figure...")

    metrics_file = DATA_ROOT / "alm" / "xl1b_metrics.json"
    metrics = load_json(metrics_file)

    fair_eval_history = metrics["fair_eval_history"]
    epochs = list(range(1, len(fair_eval_history) + 1))
    val_cer_norm = [e["val_cer_norm_fair"] for e in fair_eval_history]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

    ax.plot(epochs, val_cer_norm, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax.fill_between(epochs, val_cer_norm, alpha=0.3, color='#2E86AB')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('CER (norm)', fontsize=12, fontweight='bold')
    ax.set_title('Śabda-ALM 1.13B+LoRA: Training Trajectory (XL Corpus)\n10k clips, best checkpoint at epoch 1',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(epochs)

    # Annotate best checkpoint
    best_idx = np.argmin(val_cer_norm)
    best_epoch = epochs[best_idx]
    best_cer = val_cer_norm[best_idx]
    ax.annotate(f'Best: {best_cer:.4f}\n(epoch {best_epoch})',
                xy=(best_epoch, best_cer),
                xytext=(best_epoch + 0.3, best_cer + 0.01),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "docs" / "figures" / "01_training_trajectory.png", dpi=100, bbox_inches='tight')
    print(f"  → docs/figures/01_training_trajectory.png")
    plt.close()

def figure_sanskrit_leaderboard():
    """Figure 2: Sanskrit leaderboard - bar chart with bootstrap CI error bars."""
    print("Generating Sanskrit leaderboard figure...")

    # Per-clip records → REAL bootstrap 95% CIs (1000 resamples). No estimated error bars.
    records = load_json(DATA_ROOT / "benchmark" / "alm_vs_alm_records.json")
    wanted = ["Śabda-ALM 1.13B+LoRA XL — free decode (ours)",
              "Voxtral-Mini-3B-2507 (Mistral, open)",
              "Qwen2.5-Omni-3B Thinker (Alibaba, open)",
              "Qwen2-Audio-7B-Instruct (Alibaba, open)"]
    rng = np.random.default_rng(0)
    models, cer_norms, cis = [], [], []
    for name in wanted:
        per = np.array([r["cer_norm"] for r in records[name]])
        boot = [float(np.mean(per[rng.integers(0, len(per), len(per))])) for _ in range(1000)]
        models.append(name.replace(" — free decode (ours)", " (ours)")
                          .replace(" (Mistral, open)", "").replace(" (Alibaba, open)", ""))
        cer_norms.append(float(np.mean(per)))
        cis.append((float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))))

    errors = [(cer - ci[0], ci[1] - cer) for cer, ci in zip(cer_norms, cis)]
    errors = list(zip(*errors))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    x_pos = np.arange(len(models))
    colors = ['#A23B72' if 'Śabda' in m else '#555555' for m in models]

    bars = ax.bar(x_pos, cer_norms, yerr=errors, capsize=5,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.2, error_kw={'linewidth': 2})

    ax.set_ylabel('CER (norm)', fontsize=12, fontweight='bold')
    ax.set_title('Sanskrit Audio Recognition: Fair Comparison (58-clip Indic-Parler-TTS)\nGreedy decode, 64-byte budget, identical folded scoring',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, cer_norms)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add legend
    specialist_patch = mpatches.Patch(color='#A23B72', alpha=0.8, label='Śabda-ALM (specialist)')
    generalist_patch = mpatches.Patch(color='#555555', alpha=0.8, label='Generalist ALMs')
    ax.legend(handles=[specialist_patch, generalist_patch], loc='upper right', fontsize=10)

    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "docs" / "figures" / "02_sanskrit_leaderboard.png", dpi=100, bbox_inches='tight')
    print(f"  → docs/figures/02_sanskrit_leaderboard.png")
    plt.close()

def figure_per_clip_cers():
    """Figure 3: Per-clip CER distribution histogram (ours vs Whisper)."""
    print("Generating per-clip CER distribution figure...")

    records_file = DATA_ROOT / "benchmark" / "alm_vs_alm_records.json"
    records_data = load_json(records_file)

    # Extract CER values for Śabda-ALM 1.13B+LoRA XL and Voxtral
    sabda_cers = []
    voxtral_cers = []

    # Records are organized as {model_name: [record1, record2, ...], ...}
    for model_name, record_list in records_data.items():
        if "1.13B+LoRA XL" in model_name:
            sabda_cers = [r["cer_norm"] for r in record_list]
        elif "Voxtral" in model_name:
            voxtral_cers = [r["cer_norm"] for r in record_list]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Create histograms
    bins = np.linspace(0, 1.0, 20)
    ax.hist(voxtral_cers, bins=bins, alpha=0.6, label='Voxtral-Mini-3B', color='#555555', edgecolor='black')
    ax.hist(sabda_cers, bins=bins, alpha=0.6, label='Śabda-ALM 1.13B+LoRA', color='#A23B72', edgecolor='black')

    ax.set_xlabel('CER (norm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count (clips)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Clip CER Distribution: Śabda-ALM vs Voxtral (58 clips)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add mean lines
    sabda_mean = np.mean(sabda_cers)
    voxtral_mean = np.mean(voxtral_cers)
    ax.axvline(sabda_mean, color='#A23B72', linestyle='--', linewidth=2, label=f'Śabda mean: {sabda_mean:.4f}')
    ax.axvline(voxtral_mean, color='#555555', linestyle='--', linewidth=2, label=f'Voxtral mean: {voxtral_mean:.4f}')
    ax.legend(fontsize=10, loc='upper right')

    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "docs" / "figures" / "03_per_clip_distribution.png", dpi=100, bbox_inches='tight')
    print(f"  → docs/figures/03_per_clip_distribution.png")
    plt.close()

def figure_sphotas_lens_emergence():
    """Figure 4: Sphoṭa-Lens - meaning emergence curve by layer."""
    print("Generating Sphoṭa-Lens emergence figure...")

    emergence_file = DATA_ROOT / "sphota_lens" / "emergence_report.json"
    emergence = load_json(emergence_file)

    layers = list(range(len(emergence["decodability_by_layer"])))
    decodability = emergence["decodability_by_layer"]
    peak_layer = emergence["peak_layer"]
    chance = emergence["chance"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    ax.plot(layers, decodability, marker='o', linewidth=2, markersize=6,
            color='#2E86AB', label='Correlational decodability')
    ax.fill_between(layers, decodability, alpha=0.2, color='#2E86AB')

    # Highlight the peak
    peak_decodability = decodability[peak_layer]
    ax.scatter([peak_layer], [peak_decodability], s=150, color='red', zorder=5, label=f'Peak layer {peak_layer}')

    # Draw chance line
    ax.axhline(chance, color='gray', linestyle='--', linewidth=2, label=f'Chance: {chance:.4f}')

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Decodability (Kriyā Classification Accuracy)', fontsize=12, fontweight='bold')
    ax.set_title('Sphoṭa-Lens: Meaning Emergence by Layer\n200M core, 240 utterances, 45 verb classes',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(0, len(layers), 2))

    # Annotate the peak and show the ratio to chance
    ratio = peak_decodability / chance
    ax.annotate(f'Peak: {peak_decodability:.4f}\n({ratio:.1f}× chance)',
                xy=(peak_layer, peak_decodability),
                xytext=(peak_layer - 3, peak_decodability - 0.02),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "docs" / "figures" / "04_sphota_lens_emergence.png", dpi=100, bbox_inches='tight')
    print(f"  → docs/figures/04_sphota_lens_emergence.png")
    plt.close()

def figure_steering_uptake():
    """Figure 5: Steering workspace - uptake rates by alpha."""
    print("Generating steering uptake figure...")

    steering_file = DATA_ROOT / "alm" / "p3su_results.json"
    steering = load_json(steering_file)

    alphas = ['0.05', '0.1', '0.2', '0.4']
    uptake_rates = [steering["uptake_rates_by_alpha"][f"alpha_{a}"] for a in alphas]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)

    x_pos = np.arange(len(alphas))
    colors = ['#2E86AB' if float(a) < 0.4 else '#A23B72' for a in alphas]

    bars = ax.bar(x_pos, uptake_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Draw the random control line
    control = steering["uptake_rates_by_alpha"]["random_control"]
    ax.axhline(control, color='red', linestyle='--', linewidth=2, label=f'Random control: {control:.2f}')

    ax.set_ylabel('Uptake Rate (Top-5 Rank)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Injection Strength (α)', fontsize=12, fontweight='bold')
    ax.set_title('Steering the Workspace Band (Layers 21–23, 1.13B Model)\nConcept injection and readback, 40 val clips × 4 concepts',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'α={a}' for a in alphas], fontsize=10)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(fontsize=10, loc='upper left')

    # Add value labels on bars
    for bar, val in zip(bars, uptake_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Highlight the best (α=0.4)
    best_idx = 3
    ax.annotate(f'Best: {uptake_rates[best_idx]:.3f}\n({steering["hypothesis_H_SU1"]["ratio"]:.2f}× control)',
                xy=(best_idx, uptake_rates[best_idx]),
                xytext=(best_idx - 0.8, uptake_rates[best_idx] - 0.15),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "docs" / "figures" / "05_steering_uptake.png", dpi=100, bbox_inches='tight')
    print(f"  → docs/figures/05_steering_uptake.png")
    plt.close()

def figure_public_shrutilipi():
    """Figure 6: PUBLIC Shrutilipi-Sanskrit test leaderboard (real human speech, n=1474).

    WER with bootstrap 95% CIs straight from data/benchmark/shrutilipi_leaderboard.json
    (produced by eval_public.py --score; nothing estimated)."""
    print("Generating public Shrutilipi leaderboard figure...")
    board = load_json(DATA_ROOT / "benchmark" / "shrutilipi_leaderboard.json")["leaderboard"]

    models = [m["model"] for m in board]
    wers = [m["wer_norm"] for m in board]
    errs = list(zip(*[(m["wer_norm"] - m["wer_ci95"][0], m["wer_ci95"][1] - m["wer_norm"])
                      for m in board]))
    colors = ['#A23B72' if 'ours' in m else '#555555' for m in models]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, wers, yerr=errs, capsize=6, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=1.2, error_kw={'linewidth': 2})
    for bar, m in zip(bars, board):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{m["wer_norm"]:.3f}\n[{m["wer_ci95"][0]:.3f}, {m["wer_ci95"][1]:.3f}]',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('WER (folded)', fontsize=12, fontweight='bold')
    ax.set_title(f'PUBLIC test: Shrutilipi-Sanskrit (All India Radio, n={board[0]["n"]})\n'
                 'Free decode, identical folded scoring, bootstrap 95% CIs (1000 resamples)',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, max(m["wer_ci95"][1] for m in board) * 1.25)
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "docs" / "figures" / "06_public_shrutilipi.png", dpi=100, bbox_inches='tight')
    print("  → docs/figures/06_public_shrutilipi.png")
    plt.close()

def main():
    """Generate all figures."""
    print(f"\nGenerating figures for Śabda-ALM paper...")
    print(f"Data root: {DATA_ROOT}")
    print(f"Output root: {PROJECT_ROOT / 'docs' / 'figures'}\n")

    try:
        figure_training_trajectory()
        figure_sanskrit_leaderboard()
        figure_per_clip_cers()
        figure_sphotas_lens_emergence()
        figure_steering_uptake()
        figure_public_shrutilipi()
        print(f"\n✓ All figures generated successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Error generating figures: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
