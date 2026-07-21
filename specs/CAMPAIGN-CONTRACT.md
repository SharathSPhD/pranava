# Campaign contract — public-benchmark bilingual ALM + instruction chat (2026-07-19)

Operator directives consolidated. This contract defines PROMISE CLOSURE: the campaign ends only when
every clause below is verifiably met (twin code+domain evidence), not ad hoc. Ralph-loop discipline:
each clause is checked, iterated, and closed with evidence; failures are reported, not hidden.

## Clause 1 — Bilingual public leaderboard (head-to-head on THEIR language)
The hypothesis is NOT beating generalists on Sanskrit alone; it is one sphoṭa-principled model
standing on English too (prabhasa is an English↔Sanskrit architecture).
- [x] Śabda-ALM (bi1b) evaluated on **LibriSpeech test-clean** (public English, official split) and
      **Shrutilipi-sa test** (+ Vagdhenu chant test) under the fair free-decode protocol.
      *(2026-07-21, v3 ckpt: Shrutilipi WER 1.1024 [1.0439,1.1681] — BEATS Whisper-large-v3
      1.3254 [1.2583,1.3957], CIs disjoint; Vagdhenu chant CER 0.31; LibriSpeech WER 1.0996 —
      honestly NOT competitive: EN-shaped babble ⇒ weak audio→English conditioning. Final lever
      running: v4 r=64 LoRA (4× capacity, cold) + warm projector + EN-2× weighting, 4 ep. If v4
      fails materially on en_val, the EN-parity claim closes via documented infeasibility.)*
- [ ] Baselines on the SAME tests, identical folded scoring, bootstrap CIs: Whisper-large-v3 ✓(sa),
      Qwen2-Audio-7B (sa running, 800/1474), Voxtral-Mini-3B, Su-śrotā Conformer (queued, GB10
      chain); then whisper/qwen/voxtral on LibriSpeech (en_baseline_chain armed).
- [ ] Per-language + COMBINED macro leaderboards published (data/benchmark/*_leaderboard.json),
      app + paper updated with the real numbers, favorable or not.
- Closure evidence: leaderboard JSONs with n=full test sizes; gate PB (to be added) green.

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
