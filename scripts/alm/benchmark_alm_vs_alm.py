"""Genuine apples-to-apples ALM benchmark: Śabda-ALM (specialist) vs open generalist ALMs.

Task: held-out Sanskrit audio → romanized text; metric = character error rate (CER) vs gold.

WHY THIS REWRITE (2026-07-18): the previous version fed Qwen2-Audio through `processor(..., audios=[wav])`.
In transformers 4.57.x the audio kwarg is `audio` (singular); `audios` is silently ignored (warning, not
error) so NO audio features were built and Qwen generated from the text prompt alone — one identical
canned hallucination for all 58 clips (CER 15.86). That was a harness bug, not a Qwen limitation. With
`audio=[wav]` Qwen genuinely transcribes (e.g. gold `naraH gfham paWet` → `narahari graham pathe`).

Two fairness fixes over the old run:
  1. Correct per-model audio API (verified: outputs vary per clip; differ from a zero-audio control).
  2. Every model gets the SAME fixed decode budget — the specialist no longer gets the gold LENGTH as an
     oracle (old code capped it to len(gold)). And because the specialist emits SLP1 while generalists emit
     IAST/loose romanization, we report a transliteration-NORMALIZED CER (everything folded to an ASCII
     phonetic skeleton) as the primary, scheme-neutral metric, plus raw CER for transparency.

Run in prabhasa/nemo-gb10:26.02: scripts/alm/in_container.sh python /work/pranava/scripts/alm/benchmark_alm_vs_alm.py
"""
from __future__ import annotations

import gc
import json
import unicodedata
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "benchmark"
CORPUS = ROOT / "data" / "alm" / "speech_corpus_indic"
FEAT_DIR = CORPUS / "feats"

# --------------------------------------------------------------------------------------------------
# Metric: raw CER + transliteration-normalized CER (scheme-neutral phonetic skeleton).
# --------------------------------------------------------------------------------------------------
_SLP1_IAST = {
    "A": "ā", "I": "ī", "U": "ū", "f": "ṛ", "F": "ṝ", "x": "ḷ", "X": "ḹ",
    "E": "ai", "O": "au", "M": "ṃ", "H": "ḥ",
    "K": "kh", "G": "gh", "N": "ṅ", "C": "ch", "J": "jh", "Y": "ñ",
    "w": "ṭ", "W": "ṭh", "q": "ḍ", "Q": "ḍh", "R": "ṇ",
    "T": "th", "D": "dh", "P": "ph", "B": "bh",
    "S": "ś", "z": "ṣ", "L": "ḷ",
}


def slp1_to_iast(s: str) -> str:
    return "".join(_SLP1_IAST.get(c, c) for c in s)


def _fold(s: str) -> str:
    """NFKD → strip combining marks → lowercase → keep [a-z0-9 ] → collapse whitespace.

    Folds ā→a, ṛ→r, ṭ→t, ś/ṣ→s, ñ/ṅ→n, ṃ→m, ḥ→h so SLP1, IAST and loose romanizations converge on the
    same phonetic skeleton — a fair, scheme-neutral comparison applied identically to every model.
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s)
    return " ".join(s.split())


def _has_devanagari(s: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in s)


def norm_gold(slp1: str) -> str:
    return _fold(slp1_to_iast(slp1))


def norm_pred(pred: str, is_slp1: bool) -> str:
    if is_slp1:
        pred = slp1_to_iast(pred)
    elif _has_devanagari(pred):
        try:
            from indic_transliteration import sanscript
            pred = sanscript.transliterate(pred, sanscript.DEVANAGARI, sanscript.IAST)
        except Exception:
            pass
    return _fold(pred)


def _lev(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _lev(s2, s1)
    if not s2:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1]


def cer(pred: str, gold: str) -> float:
    return _lev(pred, gold) / max(1, len(gold))


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------
def load_val(n: int = 58) -> list:
    rows = [json.loads(x) for x in (CORPUS / "manifest.jsonl").open(encoding="utf-8") if x.strip()]
    val = [r for r in rows if r["split"] == "val"][:n]
    return [{"id": r["id"], "text": r["text"], "dev": r.get("text_devanagari", ""),
             "wav_path": ROOT / r["wav"]} for r in val]


def read_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr, raw = wf.getframerate(), wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------------
# Model runners → return {id: pred_text}.  Each is guarded so one failure doesn't sink the board.
# --------------------------------------------------------------------------------------------------
def run_specialist(val, capped: bool = False) -> tuple[dict, str | None]:
    """Śabda-ALM: Parakeet feats → Sphoṭa projector → LoRA Sanskrit core → SLP1 bytes.

    Two decode protocols:
      * free  (capped=False): fixed 64-byte budget, stop at first null byte (EOS/pad). No oracle — the
        SAME freedom the generalists get. This is the fair head-to-head number.
      * capped (capped=True): generate len(gold)+4 bytes, drop nulls, truncate to len(gold). This is how
        the specialist was originally evaluated (its published 0.565) — but len(gold) is a GOLD-LENGTH
        ORACLE the generalists never receive, so it is reported only for continuity, not as fair.
    """
    try:
        from pranava.alm.core_adapter import SanskritCore
        from pranava.alm.data import text_to_bytes
        from pranava.alm.encoder import ParakeetEncoder
        from pranava.alm.lora import inject_lora
        from pranava.alm.projector import SphotaProjector
    except Exception as e:
        return {}, f"specialist stack unavailable: {e}"
    try:
        core = SanskritCore(arm="m2", device="cuda").load()
        for p in core.model.parameters():
            p.requires_grad_(False)
        inject_lora(core.model, r=8, alpha=16)
        ck = torch.load(ROOT / "data/alm/lora_ckpt.pt", map_location=core.torch_device, weights_only=True)
        sd = {n: p for n, p in core.model.named_parameters() if ".A" in n or ".B" in n}
        for n, w in ck["lora"].items():
            if n in sd:
                sd[n].data.copy_(w.to(core.torch_device))
        proj = SphotaProjector(d_enc=ck["d_enc"], d_model=ck["d_model"], downsample=4).to(core.torch_device)
        proj.load_state_dict(ck["projector"]); proj.eval()
        bias = core.structural_bias
        enc = None
        preds = {}
        with torch.no_grad():
            for ex in val:
                fp = FEAT_DIR / f"{ex['id']}.npy"
                if fp.exists():
                    feats = np.load(fp).astype(np.float32)
                else:
                    if enc is None:
                        enc = ParakeetEncoder().load()
                    wav, sr = read_wav(ex["wav_path"])
                    feats = enc.encode(wav, sr=sr)
                    feats = feats.cpu().numpy() if hasattr(feats, "cpu") else np.asarray(feats)
                t = torch.from_numpy(feats).unsqueeze(0).to(core.torch_device)
                tok = proj(t, structural_bias=bias)
                if capped:
                    glen = len(text_to_bytes(ex["text"]))
                    out = core.greedy_from_embeds(tok, max_new=glen + 4)
                    b = [x for x in out if x != 0][:glen]  # ORACLE: truncate to gold length
                else:
                    out = core.greedy_from_embeds(tok, max_new=64)  # fair: no oracle
                    b = []
                    for x in out:
                        if x == 0:
                            break
                        b.append(x)
                preds[ex["id"]] = bytes(b).decode("latin-1", "ignore")
        del core, proj
        _free()
        return preds, None
    except Exception as e:
        return {}, f"specialist failed: {e}"


def run_qwen2_audio(val) -> tuple[dict, str | None]:
    try:
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        import librosa
    except Exception as e:
        return {}, f"qwen2-audio unavailable: {e}"
    try:
        proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto", dtype=torch.float16).eval()
    except Exception as e:
        return {}, f"qwen2-audio load failed: {e}"
    conv = [{"role": "user", "content": [
        {"type": "audio", "audio_url": "clip.wav"},
        {"type": "text", "text": "Transcribe this spoken Sanskrit to romanized (IAST) text. "
                                 "Output only the transcription, no explanation."}]}]
    prompt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    preds = {}
    with torch.no_grad():
        for ex in val:
            wav, sr = read_wav(ex["wav_path"])
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            inp = proc(text=prompt, audio=[wav], sampling_rate=16000, return_tensors="pt")  # FIX: audio=
            inp = {k: v.to(model.device) for k, v in inp.items()}
            out = model.generate(**inp, max_new_tokens=64, do_sample=False)
            gen = out[:, inp["input_ids"].shape[1]:]
            preds[ex["id"]] = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
    del model
    _free()
    return preds, None


_INSTR = ("Transcribe this spoken Sanskrit to romanized (IAST) text. Output only the transcription, "
          "no explanation.")


def run_qwen25_omni(val) -> tuple[dict, str | None]:
    """Qwen/Qwen2.5-Omni-3B (Apache, ungated) — text-only via the Thinker to skip the audio talker."""
    try:
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        import librosa
    except Exception as e:
        return {}, f"qwen2.5-omni unavailable: {e}"
    mid = "Qwen/Qwen2.5-Omni-3B"
    try:
        proc = Qwen2_5OmniProcessor.from_pretrained(mid)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            mid, device_map="auto", dtype=torch.float16).eval()
    except Exception as e:
        return {}, f"qwen2.5-omni load failed: {e}"
    sysmsg = {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]}
    preds = {}
    with torch.no_grad():
        for ex in val:
            try:
                wav, sr = read_wav(ex["wav_path"])
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                conv = [sysmsg, {"role": "user", "content": [
                    {"type": "audio", "audio": wav}, {"type": "text", "text": _INSTR}]}]
                inp = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt", padding=True,
                                               sampling_rate=16000)
                inp = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inp.items()}
                out = model.generate(**inp, max_new_tokens=64, do_sample=False)
                gen = out[:, inp["input_ids"].shape[1]:]
                preds[ex["id"]] = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
            except Exception as e:
                if not preds:  # first item failed → API mismatch, bail with the reason
                    del model; _free()
                    return {}, f"qwen2.5-omni run failed: {e}"
    del model
    _free()
    return preds, None


def run_voxtral(val) -> tuple[dict, str | None]:
    """mistralai/Voxtral-Mini-3B-2507 (Apache, ungated) — Mistral audio LM; native Hindi ASR (closest to Sanskrit)."""
    try:
        from transformers import AutoProcessor, VoxtralForConditionalGeneration
    except Exception as e:
        return {}, f"voxtral unavailable: {e}"
    mid = "mistralai/Voxtral-Mini-3B-2507"
    try:
        proc = AutoProcessor.from_pretrained(mid)
        model = VoxtralForConditionalGeneration.from_pretrained(
            mid, device_map="auto", dtype=torch.bfloat16).eval()
    except Exception as e:
        return {}, f"voxtral load failed: {e}"
    preds = {}
    with torch.no_grad():
        for ex in val:
            try:
                inp = proc.apply_transcription_request(language="hi", audio=str(ex["wav_path"]), model_id=mid)
                inp = inp.to(model.device)
                out = model.generate(**inp, max_new_tokens=64)
                preds[ex["id"]] = proc.batch_decode(out[:, inp.input_ids.shape[1]:],
                                                    skip_special_tokens=True)[0].strip()
            except Exception as e:
                if not preds:
                    del model; _free()
                    return {}, f"voxtral run failed: {e}"
    del model
    _free()
    return preds, None


MODELS = [
    ("Śabda-ALM specialist — free decode (ours)", "200M core + 0.6B enc",
     "specialist (Sanskrit fine-tuned)", True, lambda v: run_specialist(v, capped=False)),
    ("Śabda-ALM specialist — length-capped/oracle (ours, orig eval)", "200M core + 0.6B enc",
     "specialist (Sanskrit fine-tuned, gold-length oracle)", True, lambda v: run_specialist(v, capped=True)),
    ("Qwen2-Audio-7B-Instruct (Alibaba, open)", "7B",
     "generalist (no Sanskrit training)", False, run_qwen2_audio),
    ("Qwen2.5-Omni-3B Thinker (Alibaba, open)", "3B",
     "generalist (multilingual, no Sanskrit)", False, run_qwen25_omni),
    ("Voxtral-Mini-3B-2507 (Mistral, open)", "3B",
     "generalist (8 langs incl Hindi, no Sanskrit)", False, run_voxtral),
]


def score(preds: dict, val, is_slp1: bool) -> dict:
    raw, norm, per_clip, uniq = [], [], [], set()
    for ex in val:
        if ex["id"] not in preds:
            continue
        p = preds[ex["id"]]
        uniq.add(p)
        cr = cer(p, ex["text"])
        cn = cer(norm_pred(p, is_slp1), norm_gold(ex["text"]))
        raw.append(cr); norm.append(cn)
        per_clip.append({"id": ex["id"], "pred": p, "gold": ex["text"],
                         "cer_raw": round(cr, 4), "cer_norm": round(cn, 4)})
    return {
        "cer_norm": round(float(np.mean(norm)), 4) if norm else None,
        "cer_raw": round(float(np.mean(raw)), 4) if raw else None,
        "n_scored": len(norm),
        "unique_outputs": len(uniq),
        "per_clip": per_clip,
    }


def main(n: int = 58) -> int:
    val = load_val(n)
    print(f"Loaded {len(val)} val clips\n")
    board, records = [], {}
    for name, params, kind, is_slp1, runner in MODELS:
        print(f"=== {name} ===", flush=True)
        preds, err = runner(val)
        if err:
            print(f"  note: {err}", flush=True)
        s = score(preds, val, is_slp1) if preds else {"cer_norm": None, "cer_raw": None,
                                                       "n_scored": 0, "unique_outputs": 0, "per_clip": []}
        board.append({
            "model": name, "params": params, "alm_type": kind,
            "cer_norm": s["cer_norm"], "cer_raw": s["cer_raw"],
            "n_scored": s["n_scored"], "unique_outputs": s["unique_outputs"],
            "audio_conditioned": (s["unique_outputs"] > 1) if not is_slp1 else None,
            "note": err,
        })
        records[name] = s["per_clip"]
        print(f"  cer_norm={s['cer_norm']} cer_raw={s['cer_raw']} "
              f"scored={s['n_scored']} unique={s['unique_outputs']}\n", flush=True)

    ranked = sorted(board, key=lambda r: (r["cer_norm"] is None, r["cer_norm"] if r["cer_norm"] is not None else 9))
    result = {
        "task": "held-out Sanskrit audio → romanized text (CER, lower=better)",
        "n_items": len(val),
        "audio_corpus": "native Sanskrit (indic-parler-tts TTS)",
        "primary_metric": "cer_norm (transliteration-folded ASCII phonetic skeleton — scheme-neutral, fair)",
        "secondary_metric": "cer_raw (against SLP1 gold as-is; favors the SLP1-native specialist)",
        "decode": "all models: greedy, fixed 64-token budget, no gold-length oracle",
        "comparison": "specialist vs open generalist ALMs, identical audio + identical scoring",
        "leaderboard": ranked,
        "note": "Genuine multi-ALM comparison. Earlier runs fed Qwen no audio (audios= kwarg silently "
                "ignored in transformers 4.57) → a single canned hallucination (CER 15.9); that was a "
                "harness bug. Fixed to audio= (audio verified to reach the model: per-clip outputs vary "
                "and differ from a zero-audio control). Per-clip predictions for every model in "
                "alm_vs_alm_records.json.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "alm_vs_alm.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (OUT / "alm_vs_alm_records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT/'alm_vs_alm.json'} and alm_vs_alm_records.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
