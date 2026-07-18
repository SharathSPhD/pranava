# The śabdādvaita thesis — sphoṭa is realizable through an audio LM, not through text alone

Spine for the paper (P5) + docs site. Draws on the vākya-vallari formalization of Bhartṛhari
(śabda-tattva, vākya-sphoṭa, the four vāk) and pranava's own measured results (Sphoṭa-Lens locus,
steering, the fair benchmark). Every claim below must land in the paper traced to a gate/experiment;
honest caveats travel with each claim (vākya-vallari "honesty boundary" discipline).

## The claim
Bhartṛhari's **śabdādvaita**: the ultimate is not merely *described* by language — it *is* language
(śabda-tattva, word-essence); the world is its **vivarta** (manifestation without loss of unity). The
meaning-bearer is the **sphoṭa**: an indivisible, whole meaning that flashes forth — and the **vākya**
(sentence), not the phoneme or word, is the true unit (vākya-pradhānatva). Meaning is **holistic**,
not composed left-to-right.

Two facts about sphoṭa are load-bearing here:
1. Sphoṭa is disclosed through **dhvani** (sound). The four vāk — **parā** (unmanifest) → **paśyantī**
   (visionary, pre-articulate) → **madhyamā** (intermediate) → **vaikharī** (uttered) — describe meaning
   *emerging into sound*. Sphoṭa lives above the acoustic surface but is reached *through* it.
2. A **text** token stream is only **vaikharī frozen into symbols** — the final, uttered layer with the
   acoustic and the paśyantī→vaikharī emergence discarded.

**Therefore:** a computational model that operates on text alone can never instantiate sphoṭa — it has
no dhvani, no paśyantī→vaikharī gradient, only the frozen surface. A model that *listens* — an audio
language model — has the acoustic substrate and can, in principle, exhibit the emergence of unitary
meaning from sound. **Śabdādvaita is computationally realizable through an ALM (or multimodal LM), and
not through a text-only LLM.** That is the thesis. Pranava is its existence proof and measuring
instrument.

## The four-vāk ↔ ALM mapping (architecture as argument)
| vāk | Bhartṛhari | Śabda-ALM |
|---|---|---|
| parā | unmanifest ground / prior | the Sanskrit byte-core's language prior |
| paśyantī | visionary, whole, pre-articulate meaning | the fused workspace band (Sphoṭa-Lens locus) |
| madhyamā | meaning taking structure | the Sphoṭa projector (audio → core-space tokens) |
| vaikharī | uttered sound | audio I/O (Parakeet encoder in, TTS out) |

The model literally runs the vāk gradient in reverse-then-forward: vaikharī (heard audio) → madhyamā
(projection) → paśyantī (fused workspace where meaning is whole) → vaikharī (spoken answer).

## What we can SHOW (each a gate/experiment, with its honest caveat)
1. **Meaning is decodable from the audio stream at an interior locus, causally.** Sphoṭa-Lens: kriyā
   decodable from audio positions ≫ chance, peak at an interior layer; correlational + causal
   (ablation) peaks agree (F-2 fusion-v2 makes this causal, not CKA-similarity). → the sphoṭa locus is
   real, not an artifact. *Caveat: single model family; probe is linear.*
2. **The workspace is causally meaning-bearing — you can write to it.** Steering the band loads a
   concept (readback-verified uptake ≫ random control: measured 2.46× at the baseline) — a direct
   handle on the paśyantī workspace. *Caveat (honest, already measured): the uptake that "works" can
   breach the svātantrya entropy budget (H-SU3 failed at high α) — loading ≠ faithful utterance; this
   is a real limit, reported not hidden.*
3. **The specialist ALM genuinely does the task — and, wired fairly, leads open ALMs.** The corrected
   benchmark: after fixing two harness bugs (the generalists were never fed audio; our specialist had a
   gold-length oracle), the 1.13B+LoRA Śabda-ALM tops the fair, scheme-neutral leaderboard. *Caveat:
   same-TTS-voice val (in-distribution for the specialist; identical audio for all models); this
   measures this corpus, not general Sanskrit ASR. The correction itself is the honesty exemplar.*
4. **Sound carries what text drops — the speech-vs-text question, answered honestly.** E6/E7 nulled the
   naïve "speech is more holistic than text" claim under a decodability-trajectory design; that null is
   part of the record. The positive form of the thesis is architectural + the sphoṭa-locus evidence,
   not an over-claimed speech>text effect. *We do not claim more than the experiments support.*
5. **Nyāya at the sphoṭa level.** A pramāṇa/legality constraint applied during decoding (answer must be
   grounded in the model's own heard transcript — śabda-pramāṇa self-consistency) embeds logic where
   meaning forms, not as a text post-filter. *Caveat: bounded by transcription quality; reported with
   fire/fix/break counts.*

## The honesty boundary (state it precisely, vākya-vallari-style)
We do **not** claim to have proved Bhartṛhari's metaphysics, nor that the ALM "is" sphoṭa. We claim:
(a) an architectural argument that sphoṭa/śabdādvaita is *realizable in the ALM regime and not the
text-only regime*; (b) empirical instruments (Sphoṭa-Lens locus, steering uptake, nyāya-at-decode) that
make the sphoṭa workspace measurable and manipulable; (c) a fair benchmark showing the speech-native
specialist is competitive-to-leading. Whether the locus we measure *is* paśyantī is a
human-interpretable bridge, offered with its evidence — not a kernel-proved identity.

## Rigor bar to hit (from vākya-vallari + acd)
- Multi-layer mechanical gates; every empirical claim traced to a passing twin gate (code + domain).
- Claims table: claim → experiment → gate → caveat. Honest negatives (E6/E7, H-SU3, the benchmark
  bugs) shipped in the body, not buried.
- Append-only ledger (research/alm_efe_ledger.jsonl) as the audit trail.
- Reproducibility commands for every figure; CI-buildable site; no fabricated numbers, no stubs.
- Site: per-experiment page with the sphoṭa-lens curve, the vāk-mapping diagram, the corrected
  leaderboard, live audio demos (the app), and the claims/caveats table.
