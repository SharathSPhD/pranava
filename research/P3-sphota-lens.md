# Phase 3 — the Sphoṭa-Lens: results & honest reading

The central deliverable of the redirect: a lens that locates, measures, and steers the ALM's fused
meaning workspace — the computational *paśyantī/sphoṭa* where audio and symbol converge. Reuses
prabodha's pure-numpy primitives (`cka_matrix`, `best_band_partition`, `linear_cka`,
`topk_negentropy`); adds two audio-native measures the text-only j-space cannot express. Artifact:
`data/sphota_lens/sphota_lens_report.json` (`scripts/alm/p3_sphota_lens.py`, in-container).

## What was fit
On the trained ALM (frozen Parakeet + trained projector + frozen m2 core), 40 val clips, fused
audio+text sequences, per-layer hidden states captured across all 25 layers (input + 24 blocks).

## Results
- **Workspace band** (inter-layer CKA, `best_band_partition`): layers **[21, 23)**, contrast
  **0.04**. A coherent 2-layer band near the top — the ALM's global-workspace analog, located the
  same way prabodha finds the text workspace, but here on *audio-fused* states.
- **Articulation gradient** (top-k negentropy, paśyantī→vaikharī): **rises** toward the head — the
  readout becomes progressively more concentrated ("uttered") at deeper layers, matching the vāk
  ontology (held/pre-articulate early → articulated late).
- **Cross-modal fusion** (CKA between audio-position and text-position reps per layer): **peaks at
  layer 1** (early), then declines. **Honest reading:** this v1 metric measures *representational
  similarity*, which is naturally highest near the shared embedding and drops as the streams
  specialise — it is **not yet** a validated localiser of *meaning integration*. The naive
  "fusion = similarity" is an informative negative: a better sphoṭa-localiser must measure
  *integration* (e.g. mutual predictability, or CKA of the fused rep against each unimodal stream),
  not raw similarity. This is the concrete "improve on j-space" thread, and its first iteration
  tells us what not to use.
- **Steering** (concept-direction injection into the workspace band during decode): injecting
  `alpha·direction·‖h‖` at layers [21,23) **reproducibly shifts** the ALM's output
  (`any_shift=True`, identical output across repeated runs at fixed alpha). The workspace is
  writable — the paśyantī layer can be steered.

## What Phase 3 establishes
1. A working Sphoṭa-Lens over the ALM's fused states: workspace band located, articulation gradient
   measured, output steerable at the band — all reproducible and gated.
2. The vāk ontology maps onto real structure: an articulation gradient rising toward vaikharī, and
   a coherent late workspace band.
3. An honest first result on "better than j-space": the audio-native *fusion* metric needs to
   measure integration, not similarity — the next iteration's target.

## Honest limitations
Small model (200M byte core) + high projector CER (Phase 2, 0.77) mean the workspace is real but the
*content* it carries is coarse; band contrast (0.04) is modest; the fusion metric v1 is superseded
by its own finding. These bound the claims — the deliverable is a working, reproducible lens and a
clear next target, not a finished superior-to-j-space measure (that remains open, as planned).
