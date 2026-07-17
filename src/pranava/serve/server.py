"""Śabda-ALM inference server — makes the model usable over HTTP.

FastAPI service that loads the multilingual Śabda-ALM (Parakeet encoder + Sphoṭa Projector +
LoRA-adapted Sanskrit byte-core) once and serves speech→text. Runs in prabhasa/nemo-gb10 (needs
mamba + NeMo). Advances prabhasa's app philosophy (honest, explicit, container-portable) with a
clean JSON API a Vercel/Supabase frontend can call.

Endpoints:
  GET  /health        → readiness + model info
  POST /transcribe    → multipart/form-data audio file → {"text", "language", "ms"}
  GET  /              → a minimal built-in demo page (works with no frontend)

Run (in-container):
  PYTHONPATH=/work/pranava/src:/work/prabhasa-samskrutam/src \
  uvicorn pranava.serve.server:app --host 0.0.0.0 --port 8088
"""
# NB: no `from __future__ import annotations` — FastAPI must see real type objects
# (a stringized `Request` annotation is misread as a query param).
import io
import time
from pathlib import Path

import numpy as np

_STATE: dict = {}


def _load():
    import torch
    from pranava.alm.core_adapter import SanskritCore
    from pranava.alm.encoder import ParakeetEncoder
    from pranava.alm.lora import inject_lora
    from pranava.alm.projector import SphotaProjector

    root = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[3]
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    # Prefer the instruction-tuned checkpoint so the model can follow spoken tasks (transcribe /
    # identify agent / action …); fall back to the multilingual, then projector-only.
    ckpt = next((c for c in (root / "data/alm/instruct_ckpt.pt", root / "data/alm/lora_ckpt.pt")
                 if c.exists()), None)
    if ckpt is not None:
        inject_lora(core.model, r=8, alpha=16)
        blob = torch.load(ckpt, map_location=core.torch_device, weights_only=True)
        sd = {n: p for n, p in core.model.named_parameters() if ".A" in n or ".B" in n}
        for n, w in blob.get("lora", {}).items():
            if n in sd:
                sd[n].data.copy_(w.to(core.torch_device))
        proj = SphotaProjector(d_enc=blob["d_enc"], d_model=blob["d_model"], downsample=4).to(core.torch_device)
        proj.load_state_dict(blob["projector"] if "projector" in blob else blob["state_dict"])
    else:  # projector-only fallback
        blob = torch.load(root / "data/alm/projector.pt", map_location=core.torch_device, weights_only=True)
        proj = SphotaProjector(d_enc=blob["d_enc"], d_model=blob["d_model"], downsample=4).to(core.torch_device)
        proj.load_state_dict(blob["state_dict"])
    proj.eval()
    enc = ParakeetEncoder().load()

    # Speech-out (vaikharī out): the NeMo TTS head that lets the ALM *speak* its answer. Optional —
    # if it fails to load the server still serves text, and /speak reports so honestly.
    tts = None
    try:
        from pranava.alm.synth import TTSHead
        tts = TTSHead().load()
    except Exception as e:  # noqa: BLE001
        _STATE["tts_error"] = str(e)

    _STATE.update(core=core, proj=proj, enc=enc, bias=core.structural_bias, torch=torch, tts=tts,
                  loaded=True, instruction_model=(ckpt is not None and ckpt.name == "instruct_ckpt.pt"),
                  ckpt=str(ckpt.name if ckpt is not None else "projector-only"))


# the spoken tasks the instruction model understands (label → instruction text)
TASKS = {
    "transcribe": "transcribe the speech",
    "kriya": "what is the action",
    "karta": "who is the agent",
    "karma": "what is the object",
    "karana": "by what means",
    "language": "which language is this",
}


def _answer(wav: np.ndarray, sr: int, task: str = "transcribe", max_new: int = 96) -> str:
    """Audio → the model's text answer to `task` (paśyantī → text). Instruction-conditioned when the
    instruction checkpoint is loaded; otherwise a plain transcription."""
    from pranava.alm.instruct import EOS, build_prefix

    torch, core, proj, enc, bias = (_STATE[k] for k in ("torch", "core", "proj", "enc", "bias"))
    with torch.no_grad():
        frames = enc.encode(wav, sr=sr)
        tokens = proj(frames, structural_bias=bias)
        if _STATE.get("instruction_model"):
            prefix = build_prefix(core, tokens, TASKS.get(task, TASKS["transcribe"]))
            out = core.greedy_from_embeds(prefix, max_new=max_new, stop_token=EOS)
        else:  # multilingual / projector-only checkpoint: transcription only
            out = core.greedy_from_embeds(tokens, max_new=max_new)
    return bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()


def _speak(text: str):
    """Text → 16-bit PCM WAV bytes via the NeMo TTS head (vaikharī out). None if TTS unavailable."""
    import wave

    tts = _STATE.get("tts")
    if tts is None or not text:
        return None
    from pranava.alm.synth import SR

    audio = tts.say(text)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _read_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Decode arbitrary audio bytes (wav/webm/mp3) → mono float32. soundfile first, librosa (ffmpeg)
    fallback for browser webm/opus from the mic."""
    import soundfile as sf

    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav.astype(np.float32), int(sr)
    except Exception:
        import tempfile

        import librosa

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as f:
            f.write(data)
            f.flush()
            wav, sr = librosa.load(f.name, sr=16000, mono=True)
        return wav.astype(np.float32), int(sr)


def build_app():
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Śabda-ALM", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.on_event("startup")
    def _startup():
        _load()

    @app.get("/health")
    def health():
        return {"ok": _STATE.get("loaded", False), "model": "Śabda-ALM",
                "checkpoint": _STATE.get("ckpt"), "languages": ["en", "sa"],
                "instruction_model": _STATE.get("instruction_model", False),
                "speech_out": _STATE.get("tts") is not None, "tasks": sorted(TASKS),
                "tts_error": _STATE.get("tts_error")}

    def _clean(text: str) -> tuple[str, str]:
        lang = "sa" if text.startswith("[sa]") else "en" if text.startswith("[en]") else "?"
        return (text.split("]", 1)[1].strip() if "]" in text[:6] else text), lang

    @app.post("/transcribe")
    async def transcribe(request: Request):
        """Raw audio bytes in the request body → text (back-compat: task=transcribe)."""
        t0 = time.time()
        wav, sr = _read_audio(await request.body())
        text = _answer(wav, sr, "transcribe")
        clean, lang = _clean(text)
        return JSONResponse({"text": clean, "raw": text, "language": lang,
                             "ms": round((time.time() - t0) * 1000)})

    @app.post("/speak")
    async def speak(request: Request, task: str = "transcribe"):
        """Speech-to-speech: raw audio in → the model's spoken answer to `task`.

        Returns audio/wav (the model speaking its answer) with the text in headers, so a browser can
        play it back. `?task=` picks what the model is asked (transcribe / kriya / karta / …).
        """
        t0 = time.time()
        wav, sr = _read_audio(await request.body())
        text = _answer(wav, sr, task if task in TASKS else "transcribe")
        clean, lang = _clean(text)
        audio = _speak(clean)
        headers = {"X-Transcript": clean.encode("ascii", "ignore").decode() or "-",
                   "X-Language": lang, "X-Task": task,
                   "X-Ms": str(round((time.time() - t0) * 1000)),
                   "Access-Control-Expose-Headers": "X-Transcript,X-Language,X-Task,X-Ms"}
        if audio is None:  # TTS unavailable — degrade to text, honestly
            return JSONResponse({"text": clean, "language": lang, "task": task,
                                 "speech_out": False, "note": "TTS head unavailable on this server"},
                                headers=headers)
        from fastapi.responses import Response
        return Response(content=audio, media_type="audio/wav", headers=headers)

    @app.get("/", response_class=HTMLResponse)
    def demo():
        return _DEMO_HTML

    return app


_DEMO_HTML = """<!doctype html><meta charset=utf-8><title>Śabda-ALM</title>
<body style="font-family:system-ui;max-width:640px;margin:3rem auto;background:#0E1020;color:#E7E3D6">
<h1 style="color:#E8A13A">Śabda-ALM — speak, and it listens</h1>
<p>Upload a short audio clip (English or Sanskrit) and the model transcribes it through the Sanskrit
byte-core.</p>
<input type=file id=f accept="audio/*"><button onclick=go()>Transcribe</button>
<pre id=o style="background:#16182B;padding:1rem;border-radius:8px;margin-top:1rem"></pre>
<script>
async function go(){let f=document.getElementById('f').files[0];if(!f)return;
/* raw body */document.getElementById('o').textContent='…';
let r=await fetch('/transcribe',{method:'POST',body:f});let d=await r.json();
document.getElementById('o').textContent=JSON.stringify(d,null,2);}
</script></body>"""

app = build_app()
