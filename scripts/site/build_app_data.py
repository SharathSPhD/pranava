#!/usr/bin/env python3
"""
Build script for Śabda-ALM v3 web app.
Transforms raw data files into web-optimized JSON bundles.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# Directories
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
WEB_DATA_DIR = REPO_ROOT / "web" / "data"
WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("🏗️  Building Śabda-ALM web app data...")

# =============================================================================
# 1. EDITION: Vākyapadīya (linked to vakya-vallari, not bundled here)
# =============================================================================
print("\n📖 EDITION: Linked to live vakya-vallari project (https://github.com/SharathSPhD/vakya-vallari)")

# =============================================================================
# 2. SPHOTA-LENS: Emergence curves + steering uptake
# =============================================================================
print("\n🔬 LENS: Bundling Sphoṭa-Lens emergence and steering data...")

emergence_file = DATA_DIR / "sphota_lens" / "emergence_report.json"
sphota_lens_file = DATA_DIR / "sphota_lens" / "sphota_lens_report.json"

with open(emergence_file, 'r') as f:
    emergence = json.load(f)

with open(sphota_lens_file, 'r') as f:
    sphota_lens = json.load(f)

lens_bundle = {
    "emergence": emergence,
    "workspace": sphota_lens,
    "interpretation": {
        "emergence_meaning": "decodability_by_layer = per-layer linear probe accuracy for kriyā (action) from audio positions (above chance, peak at layer 13); causal_importance_by_layer shows causal contribution (via ablation)",
        "workspace_meaning": "fusion_by_layer = audio-text representation convergence (peak early); articulation_by_layer = phonetic detail concentration (rises toward output); steering shows band [21,23] is writable",
        "key_finding": "sphoṭa locus (paśyantī) at layer 13: where meaning becomes decodable from sound AND causally necessary"
    }
}

lens_output = WEB_DATA_DIR / "lens.json"
with open(lens_output, 'w', encoding='utf-8') as f:
    json.dump(lens_bundle, f, ensure_ascii=False, indent=2)
print(f"   ✓ lens.json → {os.path.getsize(lens_output)/1024:.1f} KB")

# =============================================================================
# 3. BENCHMARKS: ALM vs ALM leaderboard
# =============================================================================
print("\n🏆 BENCHMARKS: Bundling leaderboard data...")

alm_bench = DATA_DIR / "benchmark" / "alm_vs_alm.json"

with open(alm_bench, 'r') as f:
    benchmark = json.load(f)

bench_output = WEB_DATA_DIR / "benchmarks.json"
with open(bench_output, 'w', encoding='utf-8') as f:
    json.dump(benchmark, f, ensure_ascii=False, indent=2)
print(f"   ✓ benchmarks.json → {os.path.getsize(bench_output)/1024:.1f} KB")

# Check if public benchmark (Shrutilipi) exists; if so, bundle it too
shrutilipi_file = WEB_DATA_DIR / "shrutilipi_leaderboard.json"
if shrutilipi_file.exists():
    print(f"   ✓ shrutilipi_leaderboard.json found (in-progress public benchmark)")
else:
    print(f"   ⚠ shrutilipi_leaderboard.json not yet present (evaluation in progress)")

# =============================================================================
# 4. Validate all JSON
# =============================================================================
print("\n✓ Validating generated JSON files...")
for json_file in WEB_DATA_DIR.glob("*.json"):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            json.load(f)
        print(f"   ✓ {json_file.name}")
    except json.JSONDecodeError as e:
        print(f"   ✗ {json_file.name}: {e}")
        exit(1)

# =============================================================================
# 5. Summary
# =============================================================================
print("\n📊 Summary:")
total_size = sum(os.path.getsize(f) for f in WEB_DATA_DIR.glob("*.json"))
print(f"   Total data size: {total_size / (1024*1024):.2f} MB")
print(f"   Output directory: {WEB_DATA_DIR}")
print("\n✅ Build complete. Ready for deployment.")
