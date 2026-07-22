# Campaign contract — public-benchmark bilingual ALM + instruction chat (2026-07-19)

Operator directives consolidated. This contract defines PROMISE CLOSURE: the campaign ends only when
every clause below is verifiably met (twin code+domain evidence), not ad hoc. Ralph-loop discipline:
each clause is checked, iterated, and closed with evidence; failures are reported, not hidden.

## Clause 1 — Bilingual public leaderboard (head-to-head on THEIR language)
The hypothesis is NOT beating generalists on Sanskrit alone; it is one sphoṭa-principled model
standing on English too (prabhasa is an English↔Sanskrit architecture).
- [x] Śabda-ALM evaluated on **LibriSpeech test-clean** (n=2620) and **Shrutilipi-sa test** (n=1474)
      + **Vagdhenu chant** under the fair free-decode protocol.
- [x] Baselines on the SAME tests, identical folded scoring, bootstrap CIs: Whisper-large-v3,
      Qwen2-Audio-7B, Voxtral-Mini-3B on both languages. Su-śrotā EXCLUDED with cause (its .nemo
      needs a `multisoftmax` RNNTDecoder absent from our NeMo build — documented like MMS's missing
      Sanskrit adapter). data/benchmark/{shrutilipi,librispeech,combined}_leaderboard.json.
- [x] Per-language + COMBINED leaderboards published; app (web/data/) + paper (§3.1–3.2, PAPER.md
      §3.0–3.0.1, both papers + figures) updated with the real numbers, favorable or not.
- **CLAUSE 1 VERDICT (2026-07-22, honest closure):**
  - **Sanskrit board — WON:** ours WER **1.1024** [1.04,1.17] > Whisper 1.3254 > Qwen 2.2338 >
    Voxtral 2.3121. We top the real-broadcast-Sanskrit leaderboard (Whisper edges CER 0.640 vs 0.690;
    stated). Chant CER 0.31.
  - **English board — LOST (last of 4):** Voxtral 0.0295 < Whisper 0.0299 < Qwen 0.0747 << ours 1.0996.
  - **Combined (macro over the sets every system ran) — 2nd, NOT 1st:** Whisper **0.6776**, ours
    **1.1010**, Qwen 1.1542, Voxtral 1.1708. *We do not top the combined bilingual board — Whisper does.*
  - **English parity: proven INFEASIBLE under the frozen-Sanskrit-core + shared-adapter architecture**,
    with mechanism (not a shrug). FIVE training levers all pinned English val CER ≈0.80, never < v3:
    v3 baseline 0.799; v4 r=64 cold → EOS-collapse; v5 r=64 warm+EN-2× → 0.8016; v6/v6b language-tagged
    (lr 1e-4 diverged, lr 2.5e-5 stable) → 0.808. The text-only probe (core_prior_probe.json) shows the
    frozen core is *better* at English (1.87 nats/byte) than Sanskrit (2.68), yet the trained adapter
    *raises* English to 3.84 (below the no-adapter baseline) while improving Sanskrit — catastrophic
    language interference in one shared adapter. Attenuation (scale sweep) destroys the audio path, so
    there is no inference-time separation. A real fix needs per-language adapters — which abandons the
    one-model unity that was the hypothesis. **The bilingual-parity claim is therefore reported as NOT
    SUPPORTED; the Sanskrit claim stands, strong. This is honest closure (infeasibility with evidence).**
- Closure evidence: leaderboard JSONs (n=full), core_prior_probe.json + lora_interference_probe.json,
  five EFE ledger entries, paper §3.2 + Figure 7. **CLAUSE 1 CLOSED.**

## Clause 2 — Instruction-tuned bilingual CHAT (not simply Sanskrit ASR)
- [ ] Broadened bilingual instruction tuning: transcribe (en+sa real audio), translate sa↔en
      (from parallel corpora, TTS-voiced), language-ID, kāraka roles (sa) — one instruct model.
      *(build_bi_instruct.py written — Itihāsa parallel corpus, TTS-voiced; auto-runs after the
      GB10 baseline chain frees the GPU; then one bi-instruct training round on the 5090.)*
- [x] Serve: /chat endpoint — audio turn in EITHER language → instructed answer → spoken reply;
      per-turn latency reported honestly. Single-turn v1 documented as such (multi-turn = roadmap).
      *(2026-07-20: live on pranava-zeta, turn-verified end-to-end; X-Chat-Mode header labels
      single-turn honestly.)*
- [x] App: LISTEN evolved into a chat thread (mic per turn, either language, text + spoken
      answers, history, instruction chips, capability banner) — deployed. *(2026-07-20)*
- Closure evidence: live app turn-verified end-to-end (real recording → real answer), instruct
  metrics JSON, gate IC (instruction-chat) green.

## Clause 3 — Voice fidelity (standing confirmation)
Native Sanskrit audio only: indic-parler-tts fed Devanagari (synthetic), Shrutilipi/Vagdhenu (real
human). No English-voice-on-romanized synthesis anywhere in the training/eval path. SLP1 is a text
representation, never a voice.

## Clause 4 — Docs, spec, paper stay truthful and current
- [ ] specs/, research/CAMPAIGN-*.md updated at every milestone; claims tables match artifacts.
- [ ] Paper (PAPER.md + paper.tex + site) carries the bilingual public results with figures (G4),
      statuses (OPEN/falsified/constrainable) preserved.
- [ ] Memory updated for continuity.

## Loop discipline
While any clause is open: iterate (train → eval → record EFE observation → next lever), keep all
monitors/chains armed, commit each verified increment. Levers ranked if numbers disappoint:
more epochs/data (LS-360), rank-32 LoRA, length curriculum, decode repetition control, projector-only
+ frozen-LoRA ablations. STOP only when clauses are green or a clause is proven infeasible — in which
case the infeasibility is documented with evidence and the claim softened accordingly (that too is
closure, the honest kind).
