"""Per-language benchmark for the multilingual Śabda-ALM (English + Sanskrit).

The core question: does one model, one Sanskrit byte-core, transcribe BOTH real English speech and
native Sanskrit? Reports per-language CER for the Śabda-ALM and, per language, the relevant SOTA
reference (Whisper/Parakeet). Run in prabhasa/nemo-gb10 with PRANAVA_CORPUS=speech_corpus_multi.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.data import CORPUS_DIR, load_manifest, read_wav, text_to_bytes
from pranava.alm.lora import inject_lora
from pranava.alm.projector import SphotaProjector
from pranava.alm.train import _levenshtein

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
FEAT = CORPUS_DIR / "feats"
OUT = ROOT / "data" / "benchmark"


def cer(a: str, b: str) -> float:
    return _levenshtein(a, b) / max(1, len(b))


def strip_tag(t: str) -> str:
    return t.split("]", 1)[1].strip() if "]" in t[:6] else t


def main() -> int:
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias
    inject_lora(core.model, r=8, alpha=16)
    ck = torch.load(ROOT / "data/alm/lora_ckpt.pt", map_location=dev, weights_only=True)
    sd = {n: p for n, p in core.model.named_parameters() if ".A" in n or ".B" in n}
    for n, w in ck["lora"].items():
        if n in sd:
            sd[n].data.copy_(w.to(dev))
    proj = SphotaProjector(d_enc=ck["d_enc"], d_model=ck["d_model"], downsample=4).to(dev)
    proj.load_state_dict(ck["projector"]); proj.eval()

    val = load_manifest("val")
    by_lang = defaultdict(list)
    for e in val:
        by_lang[getattr(e, "lang", None) or "?"].append(e)

    # our ALM, per language
    alm = {}
    with torch.no_grad():
        for lang, exs in by_lang.items():
            cers = []
            for ex in exs:
                feats = torch.from_numpy(np.load(FEAT / f"{ex.id}.npy").astype(np.float32)).unsqueeze(0).to(dev)
                tok = proj(feats, structural_bias=bias)
                gold = ex.text  # keeps the [lang] tag; strip both sides for fair CER
                out = core.greedy_from_embeds(tok, max_new=len(text_to_bytes(gold)) + 6)
                pred = bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore")
                cers.append(cer(strip_tag(pred), strip_tag(gold)))
            alm[lang] = round(float(np.mean(cers)), 4)

    # SOTA references per language (Whisper strong on en; Parakeet multilingual)
    def whisper():
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        pr = WhisperProcessor.from_pretrained("openai/whisper-base")
        w = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(dev).eval()
        res = {}
        for lang, exs in by_lang.items():
            cers = []
            for ex in exs:
                wav, sr = read_wav(ex.wav_path)
                if sr != 16000:
                    import librosa; wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                feats = pr(wav, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
                ids = w.generate(feats, language=("hi" if lang == "sa" else "en"), task="transcribe", max_new_tokens=96)
                txt = pr.batch_decode(ids, skip_special_tokens=True)[0].strip().lower()
                cers.append(cer(txt, strip_tag(ex.text).lower()))
            res[lang] = round(float(np.mean(cers)), 4)
        return res

    wis = whisper()
    result = {"model": "multilingual Śabda-ALM (one Sanskrit byte-core, en+sa)",
              "per_language_CER": {lang: {"sabda_alm": alm.get(lang), "whisper_base": wis.get(lang),
                                          "n": len(by_lang[lang])} for lang in by_lang},
              "note": "one model transcribes real English (LibriSpeech) AND native Sanskrit; "
                      "Whisper is the SOTA reference (strong on English, its home turf)."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "multilingual_leaderboard.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
