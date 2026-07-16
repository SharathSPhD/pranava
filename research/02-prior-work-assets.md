# Prior work on the GB10 — reusable assets (survey, 2026-07-16)

Direct inspection of the sibling projects. Only what pranava can actually reuse.

## prabodha — the autoresearch loop & gate discipline (adopt wholesale)
- **ralph loop** `scripts/ralph/loop.sh <LOOP_ID>` — a re-enterable Template-Method runner:
  1 plan (read contract card + SPEC + `research/state.json`) → 2 build (in a git **worktree**
  `loop/<id>`) → 3 test (`pytest -q`) → 4 validate (GPU experiment + a **statistics tier** per
  card) → 5 close (emit `gates/gate_<LOOP>.json`, update evolution logs, squash-merge).
- **Contract cards** `contracts/L0..L21_*.md` — one per loop, define scope + gates.
- **Dual-verdict gates** `gates/gate_L*.json`: every gate has a `code_gate` AND a `domain_gate`,
  each `{verdict, evidence}`, plus recorded `deviations` and a `signoff`. **Closure is not
  automatic** — domain gate + adversarial review + sign-off required.
- **Ethos (operator, verbatim in HANDOFF.md)**: "falsification of a hypothesis is not the aim… it
  is to innovate"; keep multiple parallels alive; use "the statistical rigour of hypothesis testing
  to **prune** viable directions", not to reject H; **honest negatives are shipped results**;
  "focus on the concepts, just don't be lost in building".
- **→ pranava adopts**: the dual-gate JSON (code + domain), contract cards per milestone, and the
  ralph 5-step discipline for Pillar II loops. Reframes Sphoṭa-Bench from pure falsification to
  **exploration that uses stats as pruning shears toward novel understanding/utility**.

## pramana — the Pramāṇa validation layer (reuse the model + method)
- Published as arXiv 2604.04937 (operator = Sharath Sathish).
- **6-phase Nyāya methodology**: Saṃśaya (doubt) → Pramāṇa (evidence sources: pratyakṣa/anumāna/
  upamāna/śabda) → Pañca-avayava (5-member syllogism) → Tarka (counterfactual) → Hetvābhāsa
  (fallacy check) → Nirṇaya (ascertainment).
- **Reusable artifacts**:
  - Fine-tuned LoRA adapter `hf_upload/nyaya-llama-3b-stage0/` (Llama-3.2-3B) + `_full` variant.
  - Training data `data/training/stage_0.jsonl`, `stage_1.jsonl`; `data/vyapti_probe/`.
  - Eval results `results/stage_{0,1}_*.json`.
- **→ pranava X1**: run extracted KG claims / experiment conclusions through this 6-phase check as
  the NSM "Layer C" epistemic validator.

## Others (context)
- **jSpace** — GNW / "verbalizable representations form a global workspace" (Anthropic j-space
  reframe); the NSM "global-workspace integration" layer descends from this.
- **PWM / prayoga / ActiveCircuitDiscovery** — Pratyabhijñā world-model + mech-interp
  (transformer-lens/nnsight, CUDA env at ActiveCIrcuitDiscovery/.venv) — probing infrastructure
  reusable for Pillar II's representation probes.
- **Saṃsādhanī** — morphology works; segmenter/parser blocked (see 01-samsaadhanii-integration.md).
