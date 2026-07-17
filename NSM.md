# The Nāda-Sphoṭa Machine (NSM) — pranava as a platform and research testbed

Pranava is not one model. It is a **platform**: a stack whose layers map the four levels of *vāk*
(Bhartṛhari's *Vākyapadīya*) onto working machinery, and a **research testbed** whose gates + EFE
loop let each layer be swapped and re-measured. Multiple products fall out of the same stack.

```
  vāk level        NSM layer                              artifact (this repo)
  ─────────────    ────────────────────────────────────  ─────────────────────────────────
  vaikharī    →    Layer A · acoustic continuum (in/out)  Parakeet encoder ⇄ NeMo TTS
  madhyamā    →    Layer B · Sphoṭa Projector             src/pranava/alm/projector.py
  paśyantī    →    the sphoṭa workspace + Sphoṭa-Lens      prabhasa core + sphota_lens/
  parā        →    Śabda-Brahman (frozen prior)           prabhasa-samskrutam byte-core
  ─────────────    ────────────────────────────────────
  (over the top)   Layer C · pramāṇa validation           src/pranava/nsm/layer_c.py
```

## The layers
- **Layer A — acoustic continuum (vaikharī).** Sound in and out: a frozen Parakeet FastConformer
  encoder turns waveform into frame reps; NeMo TTS turns core text back into audio. The gross,
  articulated level — the only one the world hears.
- **Layer B — the Sphoṭa Projector (madhyamā).** A trainable conv-downsample + MLP that maps the
  acoustic continuum into the core's embedding space — the intermediate, still-forming speech. This
  is the *learned bridge*; everything downstream is frozen or LoRA-adapted.
- **The sphoṭa workspace (paśyantī).** The prabhasa Sanskrit byte-core, driven by projected audio via
  `inputs_embeds`. Meaning "flashes" here as a whole. The **Sphoṭa-Lens** (`sphota_lens/`) locates
  that flash — a validated layer 13 where kriyā becomes decodable from audio positions (correlational
  + causal peaks agree). This is the platform's *instrument*, its analogue of prabodha's j-lens but
  grounded in sound.
- **Śabda-Brahman (parā).** The frozen 200M/1B byte-core prior — the source the whole stack unfolds
  from and never overwrites.
- **Layer C — pramāṇa validation.** A Navya-Nyāya epistemic gate *after* generation: each utterance
  is audited (pratyakṣa = decode confidence, śabda = romanization legality, anumāna = non-repetition)
  and declared *ascertained* (nirṇaya) or not — a gold-free confidence/hallucination filter the
  products expose. Reuses the operator's `pramana` auditor.

## The products (one stack, many outputs)
| product | what it is | entry point |
|---|---|---|
| **Śabda-ALM API** | speech → text, multilingual (en + sa) | `src/pranava/serve/server.py` |
| **Live app** | mic/upload web client | https://sabda-alm.vercel.app |
| **Instruction model** | one audio, six tasks by prompt | `scripts/alm/train_instruct.py` |
| **Sphoṭa-Lens** | locate/measure meaning-from-sound | `src/pranava/sphota_lens/` |
| **Pramāṇa filter** | epistemic verdict on any output | `src/pranava/nsm/layer_c.py` |

## The research testbed
Every claim is a **dual-verdict gate** (`gates/check.py`, 30+ gates: code_gate + domain_gate), so a
new architecture is admitted only when it passes both machine checks. The **EFE loop**
(`src/pranava/autoresearch/alm_loop.py`) proposes the next experiment scored by expected leaderboard
gain per GPU-hour, and a re-entrant ledger records every proposed→run→observed cycle. Swapping a
layer (a bigger core, a different encoder, a new projector) is a testbed operation: re-run the gates,
read the leaderboard, keep what improves. That is what makes pranava a *bed* for surfacing new
speech-cognition architectures, not a single frozen model.
