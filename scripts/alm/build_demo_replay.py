"""Precompute the REPLAY demo artifacts served to non-admin ('user') accounts.

Runs the instruction-tuned Śabda-ALM + TTS on a handful of held-out Sanskrit clips across several
tasks, saving the input clip, the model's spoken answer (WAV), and a manifest. These are bundled as
static files under web/demo/ so the deployed app can offer a genuine speech-to-speech *replay*
(record→answer, but from stored results) to users without live GPU access. Admins/guests instead hit
the live tunnel. Run in prabhasa/nemo-gb10.

    python scripts/alm/build_demo_replay.py
"""
from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.encoder import ParakeetEncoder
from pranava.alm.instruct import EOS, build_prefix
from pranava.alm.lora import inject_lora
from pranava.alm.projector import SphotaProjector
from pranava.alm.synth import SR, TTSHead

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/alm/speech_corpus_indic"
OUT = ROOT / "web/demo"
TASKS = {"transcribe": "transcribe the speech", "kriya": "what is the action",
         "karta": "who is the agent", "karma": "what is the object"}
KARAKA = {"kriya": "kriyA", "karta": "karwA", "karma": "karma"}


def _write_wav(path: Path, audio: np.ndarray, sr: int):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def main(n_clips: int = 6) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias
    inject_lora(core.model, r=8, alpha=16); core.model.to(dev)
    blob = torch.load(ROOT / "data/alm/instruct_ckpt.pt", map_location=dev, weights_only=True)
    proj = SphotaProjector(d_enc=blob["d_enc"], d_model=blob["d_model"], downsample=4).to(dev)
    proj.load_state_dict(blob["projector"])
    lsd = {n: p for n, p in core.model.named_parameters() if ".A" in n or ".B" in n}
    for n, w in blob["lora"].items():
        if n in lsd:
            lsd[n].data.copy_(w.to(dev))
    proj.eval()
    enc = ParakeetEncoder().load()
    tts = TTSHead().load()

    rows = [json.loads(x) for x in (CORPUS / "manifest.jsonl").open()]
    val = [r for r in rows if r.get("split") == "val"][:n_clips]
    from pranava.alm.data import read_wav

    manifest = []
    for r in val:
        cid = r["id"]
        wav_in, sr = read_wav(CORPUS / f"wav/{cid}.wav")
        shutil.copy(CORPUS / f"wav/{cid}.wav", OUT / f"{cid}_in.wav")
        kmap = {role: filler for pair in r.get("karaka", []) if len(pair) == 2 for filler, role in [pair]}
        with torch.no_grad():
            frames = enc.encode(wav_in, sr=sr)
            tokens = proj(frames, structural_bias=bias)
        for task, instr in TASKS.items():
            gold = r["text"] if task == "transcribe" else kmap.get(KARAKA.get(task, ""), "")
            if task != "transcribe" and not gold:
                continue
            with torch.no_grad():
                out = core.greedy_from_embeds(build_prefix(core, tokens, instr), max_new=64, stop_token=EOS)
            answer = bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()
            _write_wav(OUT / f"{cid}_{task}_out.wav", tts.say(answer or "a"), SR)
            manifest.append({"clip_id": cid, "task": task, "gold": gold, "answer": answer,
                             "audio_in": f"demo/{cid}_in.wav", "audio_out": f"demo/{cid}_{task}_out.wav",
                             "lang": "sa"})
    (OUT / "demo.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({"clips": len(val), "artifacts": len(manifest),
                      "tasks": sorted({m["task"] for m in manifest})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
