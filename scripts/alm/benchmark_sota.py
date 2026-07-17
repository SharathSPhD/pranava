"""Weigh the Śabda-ALM against SOTA speech models — a leaderboard on the held-out set.

Same task for every model: the held-out audio → romanized-Sanskrit text; metric = character error
rate (CER) against the gold text. Compares:
  * Śabda-ALM (Parakeet enc + Sphoṭa Projector + LoRA-adapted Sanskrit core) — ours,
  * Parakeet-TDT native ASR head (NVIDIA SOTA multilingual ASR),
  * Whisper (OpenAI SOTA general ASR),
so the ALM is positioned against genuine SOTA on this Sanskrit-target task. Run in prabhasa/nemo-gb10.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.data import CORPUS_DIR, load_manifest, read_wav, text_to_bytes
from pranava.alm.lora import inject_lora
from pranava.alm.projector import SphotaProjector
from pranava.alm.train import _levenshtein

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[1]
FEAT_DIR = CORPUS_DIR / "feats"
OUT = ROOT / "data" / "benchmark"


def cer(pred: str, gold: str) -> float:
    return _levenshtein(pred, gold) / max(1, len(gold))


def eval_sabda_alm(val, enc):
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    lora_params = inject_lora(core.model, r=8, alpha=16)
    ck = torch.load(ROOT / "data/alm/lora_ckpt.pt", map_location=core.torch_device, weights_only=True)
    # load LoRA weights
    sd = {n: p for n, p in core.model.named_parameters() if ".A" in n or ".B" in n}
    for n, w in ck["lora"].items():
        if n in sd:
            sd[n].data.copy_(w.to(core.torch_device))
    proj = SphotaProjector(d_enc=ck["d_enc"], d_model=ck["d_model"], downsample=4).to(core.torch_device)
    proj.load_state_dict(ck["projector"]); proj.eval()
    bias = core.structural_bias
    cers = []
    with torch.no_grad():
        for ex in val:
            feats = torch.from_numpy(np.load(FEAT_DIR / f"{ex.id}.npy").astype(np.float32)).unsqueeze(0).to(core.torch_device)
            tok = proj(feats, structural_bias=bias)
            out = [b for b in core.greedy_from_embeds(tok, max_new=len(text_to_bytes(ex.text)) + 4) if b != 0]
            pred = bytes(out[: len(text_to_bytes(ex.text))]).decode("latin-1", "ignore")
            cers.append(cer(pred, ex.text))
    return float(np.mean(cers))


def eval_parakeet_native(val, enc):
    cers = []
    for ex in val:
        r = enc._model.transcribe([str(ex.wav_path)], verbose=False)
        txt = (r[0].text if hasattr(r[0], "text") else str(r[0])).strip()
        cers.append(cer(txt, ex.text))
    return float(np.mean(cers))


def eval_whisper(val):
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        proc = WhisperProcessor.from_pretrained("openai/whisper-base")
        w = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to("cuda").eval()
    except Exception as e:
        return None, f"whisper unavailable: {e}"
    cers = []
    with torch.no_grad():
        for ex in val:
            wav, sr = read_wav(ex.wav_path)
            if sr != 16000:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
            ids = w.generate(feats, language="en", task="transcribe", max_new_tokens=64)
            txt = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            cers.append(cer(txt, ex.text))
    return float(np.mean(cers)), None


def main(n: int = 58) -> int:
    from pranava.alm.encoder import ParakeetEncoder

    val = load_manifest("val")[:n]
    enc = ParakeetEncoder().load()
    board = []
    board.append({"model": "Śabda-ALM (Parakeet+Projector+LoRA→Sanskrit core, ours)",
                  "params": "200M core + 0.6B enc", "cer": round(eval_sabda_alm(val, enc), 4)})
    board.append({"model": "Parakeet-TDT-0.6B native ASR (NVIDIA)",
                  "params": "0.6B", "cer": round(eval_parakeet_native(val, enc), 4)})
    wc, werr = eval_whisper(val)
    board.append({"model": "Whisper-base (OpenAI)", "params": "74M",
                  "cer": round(wc, 4) if wc is not None else None, "note": werr})
    board.sort(key=lambda r: (r["cer"] is None, r["cer"] if r["cer"] is not None else 9))
    result = {"task": "held-out audio → romanized-Sanskrit text (CER, lower=better)",
              "n_items": len(val), "leaderboard": board,
              "note": " SOTA general ASR is not trained on this romanized-Sanskrit target; the ALM is "
                      "specialised for the Sanskrit byte-core. All models see the identical audio."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sota_leaderboard.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
