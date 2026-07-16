# Pre-registration — H-HOLISM (E2)

**Committed before any E2 results are computed.** Git history proves ordering. Ethos (prabodha):
the aim is to *learn where the sphoṭa analogy holds*, not to confirm it. A null is a shipped result.

## Question
Does decodable sentence-meaning emerge more **holistically** — late and concentrated, a "flash" —
in continuous **speech** representations than in discrete **text** representations, as Bhartṛhari's
*pratibhā* thesis predicts (Vākyapadīya 2.143: sentence-meaning is a *distinct* cognition arising
when word-meanings are grasped together, not their running sum)? And is this holism greater for
sentences whose meaning genuinely resolves late?

## Materials
- **Stimuli**: the 288 controlled items from E1 (`data/stimuli/manifest.jsonl`), balanced
  early (168) / late (120), across structures {canonical, garden_path, verb_final, verb_first},
  11 meaning labels, exact disambiguation indices.
- **Speech model**: WavLM-base frames (`pranava.speech.harness`), a chosen fixed layer.
- **Text model**: a text LM's per-token hidden states (GPT-2 small), same fixed-layer policy.

## Procedure
1. For each item, extract per-position representations: speech = per-frame hidden states;
   text = per-token hidden states of the written sentence.
2. Build **cumulative representations** at relative positions t ∈ {0.1, 0.2, …, 1.0}: mean-pool all
   frames/tokens up to ⌈t·N⌉.
3. Train a **linear probe** (multinomial logistic regression) to predict `meaning_label` from the
   *full-utterance* representation, evaluated with **template-grouped K-fold CV** (no template
   appears in both train and test → no lexical leakage). Fit the probe once per (modality, seed).
4. Apply the probe to each cumulative representation → decodability accuracy vs position curve,
   per item, per modality.

## Primary metric — Holism Index (HI)
For an accuracy-vs-position curve a(t):
  HI = clip( (a(1.0) − a(0.8)) / (a(1.0) − a(0.1) + ε), 0, 1 )
High HI ⇒ most decodability gain arrives in the last 20% (late flash / holistic).
Low HI ⇒ gradual accrual. HI is computed per item (using the group-out probe) then averaged.

## Confirmatory hypotheses (directional)
- **P1**: HI(late items) > HI(early items), within each modality. Null: HI_late = HI_early.
- **P2 (key)**: HI_speech(late) > HI_text(late). Null: HI_speech = HI_text.

## Statistics (fixed in advance)
- Effect = mean HI difference. **Bootstrap 95% CI** over items, ≥2000 resamples.
- Probe trained with ≥3 seeds; report mean ± sd across seeds.
- Multiple-comparison control: **Holm** across the 2 primary tests.
- A hypothesis is *supported* iff its bootstrap CI excludes 0 in the predicted direction after Holm.
- **Stopping rule**: fixed dataset (288 items). No data-dependent stopping. No item exclusion
  except items that fail to encode (logged).

## Exploratory (NOT confirmatory; reported as such)
Per-layer HI sweep; garden_path vs verb_final; verb_first early-resolution; alternative pooling
(last-token vs mean); alignment of the HI jump with the annotated disambiguation index.

## Threats to validity (logged, not hidden)
Synthetic single-speaker TTS artifacts; small closed vocabulary; word→frame alignment is
approximate; "cumulative mean-pool" is one operationalization of "grasped so far"; linear-probe
capacity. Speech and text tokenize at different granularities — relative position t normalizes but
does not equate them; this is a stated interpretive limit, examined in the exploratory layer.

## Decision table
| Outcome | Reading |
|---|---|
| P1 & P2 supported | speech resolves meaning more holistically & late — sphoṭa analogy holds here |
| P1 only | late items are holistic in both modalities; no speech advantage |
| P2 only | speech more holistic regardless of resolution timing |
| neither | no holism signal under this operationalization — analogy does not transfer as posed |

All four are publishable. We prune the direction, we do not force the result.
