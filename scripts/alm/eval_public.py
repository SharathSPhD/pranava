"""PUBLIC bilingual benchmark harness — Śabda-ALM head-to-head on English AND Sanskrit, one metric.

Operator hypothesis: don't just beat generalists where they're untrained (Sanskrit) — stand on their
home ground (English). Datasets (all public, official/pre-registered splits, tests untouched):
  * librispeech  — LibriSpeech test-clean (THE standard public English ASR benchmark)
  * shrutilipi   — amithm3/shrutilipi sa/test (real human Sanskrit, broadcast)
  * vagdhenu     — prathoshap/vagdhenu-data session-held-out test (real human Sanskrit chant)

Scoring: identical fold for every system per dataset — English: ASCII fold (lowercase, strip
non-alnum); Sanskrit: Devanagari-junk-strip → IAST → ASCII fold (SLP1 outputs transliterated first).
WER + CER with 1,000-resample bootstrap 95% CIs. Per-language leaderboards + a COMBINED (macro-avg
across datasets) leaderboard — models missing a language score 1.0-floor… no: models missing a dataset
are shown with a gap and excluded from combined (documented), no imputation.

  --dataset X --model Y   run one model on one dataset → records JSON
  --dataset X --score     score one dataset's records
  --combined              build the combined leaderboard from all scored datasets

Models: sabda (5090) | whisper | qwen | voxtral | sushrota (Sanskrit-only, GB10).
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]

DATASETS = {
    "shrutilipi": {"dir": ROOT / "data/alm/shrutilipi_sa", "gold_slp1": True, "lang": "sa",
                   "whisper_lang": "sanskrit", "voxtral_lang": "hi",
                   "qwen_instr": "Transcribe this Sanskrit speech in Devanagari script. Output only the transcription.",
                   "sabda_instr": "transcribe the speech", "max_new": 448},
    "vagdhenu": {"dir": ROOT / "data/alm/vagdhenu", "gold_slp1": True, "lang": "sa",
                 "whisper_lang": "sanskrit", "voxtral_lang": "hi",
                 "qwen_instr": "Transcribe this Sanskrit chant in Devanagari script. Output only the transcription.",
                 "sabda_instr": "transcribe the speech", "max_new": 448},
    "librispeech": {"dir": ROOT / "data/alm/librispeech", "gold_slp1": False, "lang": "en",
                    "whisper_lang": "english", "voxtral_lang": "en",
                    "qwen_instr": "Transcribe this English speech. Output only the transcription.",
                    "sabda_instr": "transcribe the speech", "max_new": 448},
}

_NOT_DEVA = re.compile(r"[^ऀ-ॿ\s]")


def _has_deva(s: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in s)


def fold(text: str, is_slp1: bool) -> str:
    from indic_transliteration import sanscript
    text = text or ""
    if is_slp1:
        text = sanscript.transliterate(text, sanscript.SLP1, sanscript.IAST)
    elif _has_deva(text):
        text = unicodedata.normalize("NFC", text)
        text = _NOT_DEVA.sub(" ", text)
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


def rows_of(ds: str, split: str = "test") -> list[dict]:
    d = DATASETS[ds]
    rows = [json.loads(x) for x in (d["dir"] / "manifest.jsonl").open(encoding="utf-8") if x.strip()]
    return [r for r in rows if r["split"] == split]


def rec_dir(ds: str) -> Path:
    return ROOT / f"data/benchmark/{ds}_records"


def _read_wav16(path: Path):
    import librosa
    import soundfile as sf
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def _save(ds: str, model_name: str, pred_slp1: bool, per_clip: list[dict]):
    rd = rec_dir(ds)
    rd.mkdir(parents=True, exist_ok=True)
    p = rd / f"{re.sub(r'[^A-Za-z0-9_-]', '_', model_name)}.json"
    p.write_text(json.dumps({"model": model_name, "is_slp1": pred_slp1, "per_clip": per_clip},
                            indent=2, ensure_ascii=False))
    print(f"wrote {p} ({len(per_clip)} clips)")


# ---------------------------------------- model runners ----------------------------------------
def run_sabda(ds: str, rows):
    import torch
    from pranava.alm.instruct import EOS, build_prefix
    from pranava.alm.megatron_core import Megatron1BCore
    from pranava.alm.megatron_lora import inject_megatron_lora, load_megatron_lora
    from pranava.alm.projector import SphotaProjector
    from pranava.alm.encoder import ParakeetEncoder

    cfg = DATASETS[ds]
    ck = next((p for p in (ROOT / "data/alm/bi1b_ckpt.pt", ROOT / "data/alm/sh1b_ckpt.pt") if p.exists()))
    blob = torch.load(ck, map_location="cpu", weights_only=True)
    print(json.dumps({"ckpt": ck.name, "epoch": blob.get("epoch"), "val": blob.get("val_cer_norm_fair")}))
    core = Megatron1BCore().load()
    for p in core._model.parameters():
        p.requires_grad_(False)
    inject_megatron_lora(core._model, r=int(blob.get("r", 16)))
    load_megatron_lora(core._model, blob["lora"])
    proj = SphotaProjector(d_enc=blob["d_enc"], d_model=blob["d_model"], downsample=4).to(core.torch_device)
    proj.load_state_dict(blob["projector"]); proj.eval()
    bias = core.structural_bias
    feats_dir = cfg["dir"] / "feats"
    enc = None
    per = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            try:
                fp = feats_dir / f"{r['id']}.npy"
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
                audio = proj(t, structural_bias=bias)
                prefix = build_prefix(core, audio, cfg["sabda_instr"])
                out = core.greedy_from_embeds(prefix, max_new=cfg["max_new"], stop_token=EOS)
                pred = bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()
            except Exception as e:
                per.append({"id": r["id"], "pred": "", "gold": r["text"], "note": f"input_error: {e}"})
                continue
            per.append({"id": r["id"], "pred": pred, "gold": r["text"]})
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(rows)}", flush=True)
    # our model emits SLP1 for sa targets, plain ASCII for en targets
    _save(ds, "Sabda-ALM 1.13B+LoRA (ours)", DATASETS[ds]["gold_slp1"], per)


def run_whisper(ds: str, rows):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    cfg = DATASETS[ds]
    mid = "openai/whisper-large-v3"
    proc = WhisperProcessor.from_pretrained(mid)
    model = WhisperForConditionalGeneration.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()
    per = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            wav = _read_wav16(ROOT / r["wav"])
            feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
            ids = model.generate(feats, language=cfg["whisper_lang"], task="transcribe", max_new_tokens=400)
            pred = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            per.append({"id": r["id"], "pred": pred, "gold": r["text"]})
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(rows)}", flush=True)
    _save(ds, "Whisper-large-v3 (OpenAI)", False, per)


def run_qwen(ds: str, rows):
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    cfg = DATASETS[ds]
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto", dtype=torch.float16).eval()
    conv = [{"role": "user", "content": [
        {"type": "audio", "audio_url": "clip.wav"},
        {"type": "text", "text": cfg["qwen_instr"]}]}]
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
    _save(ds, "Qwen2-Audio-7B-Instruct (Alibaba)", False, per)


def run_voxtral(ds: str, rows):
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    cfg = DATASETS[ds]
    mid = "mistralai/Voxtral-Mini-3B-2507"
    proc = AutoProcessor.from_pretrained(mid)
    model = VoxtralForConditionalGeneration.from_pretrained(mid, device_map="auto", dtype=torch.bfloat16).eval()
    per = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            inp = proc.apply_transcription_request(language=cfg["voxtral_lang"], audio=str(ROOT / r["wav"]), model_id=mid)
            inp = inp.to(model.device)
            out = model.generate(**inp, max_new_tokens=400)
            pred = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            per.append({"id": r["id"], "pred": pred, "gold": r["text"]})
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(rows)}", flush=True)
    _save(ds, "Voxtral-Mini-3B-2507 (Mistral)", False, per)


def run_sushrota(ds: str, rows):
    if DATASETS[ds]["lang"] != "sa":
        raise SystemExit("sushrota is a Sanskrit-only model — not applicable to this dataset")
    import glob
    import nemo.collections.asr as nemo_asr
    cand = sorted(glob.glob(str(ROOT / "data/models/sushrota/**/*.nemo"), recursive=True))
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
    _save(ds, "Sushrota Sanskrit-ASR ConformerCTC (public specialist)", False, per)


# ---------------------------------------- scoring ----------------------------------------
def score(ds: str):
    cfg = DATASETS[ds]
    rng = np.random.default_rng(0)
    board = []
    for p in sorted(rec_dir(ds).glob("*.json")):
        d = json.loads(p.read_text())
        per, pred_slp1 = d["per_clip"], d["is_slp1"]
        cers, wers = [], []
        for r in per:
            pf, gf = fold(r["pred"], pred_slp1), fold(r["gold"], cfg["gold_slp1"])
            cers.append(_lev(pf, gf) / max(1, len(gf)))
            pw, gw = pf.split(), gf.split()
            wers.append(_lev(pw, gw) / max(1, len(gw)))
        cers, wers = np.array(cers), np.array(wers)
        bc = [float(np.mean(cers[rng.integers(0, len(cers), len(cers))])) for _ in range(1000)]
        bw = [float(np.mean(wers[rng.integers(0, len(wers), len(wers))])) for _ in range(1000)]
        board.append({"model": d["model"], "n": len(per),
                      "wer_norm": round(float(np.mean(wers)), 4),
                      "wer_ci95": [round(float(np.percentile(bw, 2.5)), 4), round(float(np.percentile(bw, 97.5)), 4)],
                      "cer_norm": round(float(np.mean(cers)), 4),
                      "cer_ci95": [round(float(np.percentile(bc, 2.5)), 4), round(float(np.percentile(bc, 97.5)), 4)]})
    board.sort(key=lambda r: r["wer_norm"])
    out = ROOT / f"data/benchmark/{ds}_leaderboard.json"
    res = {"dataset": ds, "lang": cfg["lang"], "n_test": len(rows_of(ds)),
           "protocol": "identical fold per dataset; free decode; bootstrap 95% CIs (1000)",
           "leaderboard": board}
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


def combined():
    """Macro-average across scored datasets; a model appears only if it ran on ALL sa+en sets it can
    (sushrota is Sanskrit-only by design and is shown per-language, excluded from combined)."""
    boards = {}
    for ds in DATASETS:
        p = ROOT / f"data/benchmark/{ds}_leaderboard.json"
        if p.exists():
            boards[ds] = {r["model"]: r for r in json.loads(p.read_text())["leaderboard"]}
    models = set().union(*[set(b) for b in boards.values()]) if boards else set()
    rows = []
    for m in models:
        per_ds = {ds: boards[ds].get(m) for ds in boards}
        have = {ds: v for ds, v in per_ds.items() if v}
        entry = {"model": m,
                 **{f"wer_{ds}": (v["wer_norm"] if v else None) for ds, v in per_ds.items()},
                 "datasets_covered": sorted(have)}
        if len(have) == len(boards):  # full coverage → combined macro-avg
            entry["wer_combined_macro"] = round(float(np.mean([v["wer_norm"] for v in have.values()])), 4)
        rows.append(entry)
    rows.sort(key=lambda r: (r.get("wer_combined_macro") is None, r.get("wer_combined_macro", 9)))
    res = {"note": "combined = macro-average WER over all scored public test sets; models without full "
                   "coverage shown per-language only (no imputation)",
           "datasets": sorted(boards), "leaderboard": rows}
    (ROOT / "data/benchmark/combined_leaderboard.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


RUNNERS = {"sabda": run_sabda, "whisper": run_whisper, "qwen": run_qwen,
           "voxtral": run_voxtral, "sushrota": run_sushrota}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="shrutilipi")
    ap.add_argument("--model", choices=sorted(RUNNERS))
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--combined", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.combined:
        combined()
    elif a.score:
        score(a.dataset)
    else:
        rows = rows_of(a.dataset)
        if a.limit:
            rows = rows[: a.limit]
        print(json.dumps({"dataset": a.dataset, "model": a.model, "n_test": len(rows)}), flush=True)
        RUNNERS[a.model](a.dataset, rows)
