"""PUBLIC-BENCHMARK eval: Shrutilipi-sa TEST split — Śabda-ALM vs open ASR/ALM baselines, one metric.

The test split is the public amithm3/shrutilipi sa/test (REAL human Sanskrit, never filtered, never
seen in training). Every system is scored IDENTICALLY: output text → drop non-Devanagari junk (for
Devanagari outputs) → transliterate to the folded ASCII phonetic space (SLP1/Devanagari→IAST→fold) →
CER and WER vs the identically-folded reference, with 1,000-resample bootstrap 95% CIs.

Usage (modular so each model runs where it fits):
  --model sabda     (RTX 5090, nemo-5090: loads sh1b_ckpt)          → records JSON
  --model whisper   (GB10 container: openai/whisper-large-v3, language sanskrit)
  --model mms       (GB10: facebook/mms-1b-all, target_lang san)
  --model qwen      (GB10: Qwen2-Audio-7B-Instruct, audio= fix)
  --model voxtral   (GB10: Voxtral-Mini-3B, transcription mode)
  --score           (host: merge all records → leaderboard + CIs → data/benchmark/shrutilipi_sa_leaderboard.json)
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
SH = ROOT / "data/alm/shrutilipi_sa"
REC_DIR = ROOT / "data/benchmark/shrutilipi_records"
OUT = ROOT / "data/benchmark/shrutilipi_sa_leaderboard.json"

_SLP1_IAST = {"A": "ā", "I": "ī", "U": "ū", "f": "ṛ", "F": "ṝ", "x": "ḷ", "X": "ḹ", "E": "ai",
              "O": "au", "M": "ṃ", "H": "ḥ", "K": "kh", "G": "gh", "N": "ṅ", "C": "ch", "J": "jh",
              "Y": "ñ", "w": "ṭ", "W": "ṭh", "q": "ḍ", "Q": "ḍh", "R": "ṇ", "T": "th", "D": "dh",
              "P": "ph", "B": "bh", "S": "ś", "z": "ṣ", "L": "ḷ"}
_NOT_DEVA = re.compile(r"[^ऀ-ॿ\s]")


def _has_deva(s: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in s)


def fold(text: str, is_slp1: bool) -> str:
    """One scoring space for every system (mirrors the corpus prep cleaning exactly)."""
    from indic_transliteration import sanscript
    text = text or ""
    if is_slp1:
        text = sanscript.transliterate(text, sanscript.SLP1, sanscript.IAST)
    elif _has_deva(text):
        text = unicodedata.normalize("NFC", text)
        text = _NOT_DEVA.sub(" ", text)  # same junk-stripping as the reference prep
        text = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
    s = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = "".join(c if (("a" <= c <= "z") or ("0" <= c <= "9") or c.isspace()) else " " for c in s)
    return " ".join(s.split())


def _lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        cur = [i + 1]
        for j, c2 in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1]


def test_rows() -> list[dict]:
    rows = [json.loads(x) for x in (SH / "manifest.jsonl").open(encoding="utf-8") if x.strip()]
    return [r for r in rows if r["split"] == "test"]


def _read_wav16(path: Path):
    import soundfile as sf
    import librosa
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def _save(model_name: str, is_slp1: bool, per_clip: list[dict]):
    REC_DIR.mkdir(parents=True, exist_ok=True)
    p = REC_DIR / f"{re.sub(r'[^A-Za-z0-9_-]', '_', model_name)}.json"
    p.write_text(json.dumps({"model": model_name, "is_slp1": is_slp1, "per_clip": per_clip},
                            indent=2, ensure_ascii=False))
    print(f"wrote {p} ({len(per_clip)} clips)")


# ------------------------------------------------ models ------------------------------------------------
def run_sabda(rows):
    import torch
    from pranava.alm.instruct import EOS, build_prefix
    from pranava.alm.megatron_core import Megatron1BCore
    from pranava.alm.megatron_lora import inject_megatron_lora, load_megatron_lora
    from pranava.alm.projector import SphotaProjector

    blob = torch.load(ROOT / "data/alm/sh1b_ckpt.pt", map_location="cpu", weights_only=True)
    print(json.dumps({"ckpt_epoch": blob.get("epoch"), "ckpt_val_cer": blob.get("val_cer_norm_fair")}))
    core = Megatron1BCore().load()
    for p in core._model.parameters():
        p.requires_grad_(False)
    inject_megatron_lora(core._model, r=int(blob.get("r", 16)))
    load_megatron_lora(core._model, blob["lora"])
    proj = SphotaProjector(d_enc=blob["d_enc"], d_model=blob["d_model"], downsample=4).to(core.torch_device)
    proj.load_state_dict(blob["projector"]); proj.eval()
    bias = core.structural_bias
    feats_dir = SH / "feats"
    from pranava.alm.encoder import ParakeetEncoder
    enc = None
    per = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            fp = feats_dir / f"{r['id']}.npy"
            try:
                if fp.exists():
                    t = torch.from_numpy(np.load(fp).astype(np.float32)).unsqueeze(0).to(core.torch_device)
                else:
                    if enc is None:
                        enc = ParakeetEncoder().load()
                    wav = _read_wav16(ROOT / r["wav"])
                    t = enc.encode(wav, sr=16000)
                    if not torch.is_tensor(t):
                        t = torch.from_numpy(np.asarray(t))
                    if t.dim() == 2:
                        t = t.unsqueeze(0)
                    t = t.to(core.torch_device)
            except Exception as e:  # defective clip (e.g. zero-length audio): honest full-error penalty
                per.append({"id": r["id"], "pred": "", "gold": r["text"], "note": f"input_error: {e}"})
                continue
            audio = proj(t, structural_bias=bias)
            prefix = build_prefix(core, audio, "transcribe the speech")
            out = core.greedy_from_embeds(prefix, max_new=448, stop_token=EOS)
            pred = bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()
            per.append({"id": r["id"], "pred": pred, "gold": r["text"]})
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(rows)}", flush=True)
    _save("Sabda-ALM 1.13B+LoRA (ours)", True, per)


def run_whisper(rows):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    mid = "openai/whisper-large-v3"
    proc = WhisperProcessor.from_pretrained(mid)
    model = WhisperForConditionalGeneration.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()
    per = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            wav = _read_wav16(ROOT / r["wav"])
            feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
            ids = model.generate(feats, language="sanskrit", task="transcribe", max_new_tokens=400)
            pred = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            per.append({"id": r["id"], "pred": pred, "gold": r["text"]})
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(rows)}", flush=True)
    _save("Whisper-large-v3 (OpenAI)", False, per)


def run_mms(rows):
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    mid = "facebook/mms-1b-all"
    proc = AutoProcessor.from_pretrained(mid, target_lang="san")
    model = Wav2Vec2ForCTC.from_pretrained(mid, target_lang="san", ignore_mismatched_sizes=True).to("cuda").eval()
    per = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            wav = _read_wav16(ROOT / r["wav"])
            inp = proc(wav, sampling_rate=16000, return_tensors="pt").to("cuda")
            logits = model(**inp).logits
            ids = torch.argmax(logits, dim=-1)
            pred = proc.batch_decode(ids)[0].strip()
            per.append({"id": r["id"], "pred": pred, "gold": r["text"]})
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(rows)}", flush=True)
    _save("MMS-1B-all san (Meta)", False, per)


def run_qwen(rows):
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto", dtype=torch.float16).eval()
    conv = [{"role": "user", "content": [
        {"type": "audio", "audio_url": "clip.wav"},
        {"type": "text", "text": "Transcribe this Sanskrit speech in Devanagari script. "
                                 "Output only the transcription."}]}]
    prompt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    per = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            wav = _read_wav16(ROOT / r["wav"])
            inp = proc(text=prompt, audio=[wav], sampling_rate=16000, return_tensors="pt")
            inp = {k: v.to(model.device) for k, v in inp.items()}
            out = model.generate(**inp, max_new_tokens=400, do_sample=False)
            gen = out[:, inp["input_ids"].shape[1]:]
            pred = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
            per.append({"id": r["id"], "pred": pred, "gold": r["text"]})
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(rows)}", flush=True)
    _save("Qwen2-Audio-7B-Instruct (Alibaba)", False, per)


def run_voxtral(rows):
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    mid = "mistralai/Voxtral-Mini-3B-2507"
    proc = AutoProcessor.from_pretrained(mid)
    model = VoxtralForConditionalGeneration.from_pretrained(mid, device_map="auto", dtype=torch.bfloat16).eval()
    per = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            inp = proc.apply_transcription_request(language="hi", audio=str(ROOT / r["wav"]), model_id=mid)
            inp = inp.to(model.device)
            out = model.generate(**inp, max_new_tokens=400)
            pred = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            per.append({"id": r["id"], "pred": pred, "gold": r["text"]})
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(rows)}", flush=True)
    _save("Voxtral-Mini-3B-2507 (Mistral)", False, per)


def run_sushrota(rows):
    """prathoshap/sushrota-sanskrit-asr — the strongest PUBLIC dedicated Sanskrit ASR (NeMo
    Conformer-CTC, IndicConformer-based; Su-śrotā project). Specialist baseline, not a generalist."""
    import glob
    import nemo.collections.asr as nemo_asr
    cand = sorted(glob.glob(str(ROOT / "data/models/sushrota/**/*.nemo"), recursive=True))
    if not cand:
        raise SystemExit("no .nemo checkpoint under data/models/sushrota")
    model = nemo_asr.models.ASRModel.restore_from(cand[0], map_location="cuda").eval()
    per = []
    B = 16
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        paths = [str(ROOT / r["wav"]) for r in chunk]
        try:
            hyps = model.transcribe(paths, batch_size=len(paths), verbose=False)
        except TypeError:
            hyps = model.transcribe(paths, verbose=False)
        for r, h in zip(chunk, hyps):
            pred = h.text if hasattr(h, "text") else (h[0] if isinstance(h, (list, tuple)) else str(h))
            per.append({"id": r["id"], "pred": str(pred).strip(), "gold": r["text"]})
        print(f"  {min(i + B, len(rows))}/{len(rows)}", flush=True)
    _save("Sushrota Sanskrit-ASR ConformerCTC (public specialist)", False, per)


# ------------------------------------------------ scoring ------------------------------------------------
def score():
    rng = np.random.default_rng(0)
    board = []
    for p in sorted(REC_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        per, slp1 = d["per_clip"], d["is_slp1"]
        cers, wers = [], []
        for r in per:
            pf, gf = fold(r["pred"], slp1), fold(r["gold"], True)
            cers.append(_lev(pf, gf) / max(1, len(gf)))
            pw, gw = pf.split(), gf.split()
            wers.append(_lev(pw, gw) / max(1, len(gw)))
        cers, wers = np.array(cers), np.array(wers)
        boots_c = [float(np.mean(cers[rng.integers(0, len(cers), len(cers))])) for _ in range(1000)]
        boots_w = [float(np.mean(wers[rng.integers(0, len(wers), len(wers))])) for _ in range(1000)]
        board.append({
            "model": d["model"], "n": len(per),
            "cer_norm": round(float(np.mean(cers)), 4),
            "cer_ci95": [round(float(np.percentile(boots_c, 2.5)), 4),
                         round(float(np.percentile(boots_c, 97.5)), 4)],
            "wer_norm": round(float(np.mean(wers)), 4),
            "wer_ci95": [round(float(np.percentile(boots_w, 2.5)), 4),
                         round(float(np.percentile(boots_w, 97.5)), 4)],
        })
    board.sort(key=lambda r: r["wer_norm"])
    res = {"benchmark": "Shrutilipi-Sanskrit PUBLIC test split (amithm3/shrutilipi sa/test — REAL human "
                        "Sanskrit, All India Radio; AI4Bharat)",
           "protocol": "identical folded scoring space for every system (Devanagari junk-strip → IAST → "
                       "ASCII fold); free decode; bootstrap 95% CIs (1000 resamples)",
           "leaderboard": board}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


RUNNERS = {"sabda": run_sabda, "whisper": run_whisper, "mms": run_mms,
           "qwen": run_qwen, "voxtral": run_voxtral, "sushrota": run_sushrota}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(RUNNERS))
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap test rows")
    a = ap.parse_args()
    if a.score:
        score()
    else:
        rows = test_rows()
        if a.limit:
            rows = rows[: a.limit]
        print(json.dumps({"model": a.model, "n_test": len(rows)}), flush=True)
        RUNNERS[a.model](rows)
