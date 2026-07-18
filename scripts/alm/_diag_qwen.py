"""DIAGNOSTIC (not the benchmark): find where audio is dropped in the Qwen2-Audio pipeline.

Runs 3 different clips + a zeroed-audio control and instruments every boundary:
  transformers version → chat-template output → processor inputs → generation.
If audio is truly conditioned on, the 3 real clips differ from each other AND from the zero control.
"""
from __future__ import annotations
import inspect, json, wave
from pathlib import Path
import numpy as np, torch, transformers

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
print("transformers:", transformers.__version__, "| torch:", torch.__version__)

from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        sr, raw = wf.getframerate(), wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr

rows = [json.loads(l) for l in (ROOT/"data/alm/speech_corpus_indic/manifest.jsonl").open() if l.strip()]
val = [r for r in rows if r["split"] == "val"][:3]

proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
print("\n=== processor.__call__ signature ===")
print(list(inspect.signature(proc.__call__).parameters))
print("feature_extractor.sampling_rate:", getattr(proc.feature_extractor, "sampling_rate", "?"))

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto", torch_dtype=torch.float16).eval()

def build_prompt():
    conv = [{"role": "user", "content": [
        {"type": "audio", "audio_url": "clip.wav"},
        {"type": "text", "text": "Transcribe this spoken Sanskrit to romanized (IAST/SLP1) text. Output only the transcription."}]}]
    return proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

prompt = build_prompt()
print("\n=== chat-template prompt ===")
print(repr(prompt))
print("contains <|AUDIO|>:", "<|AUDIO|>" in prompt or "<|audio_bos|>" in prompt)

def run(wav, sr, tag):
    import librosa
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    inputs = proc(text=prompt, audio=[wav], sampling_rate=16000, return_tensors="pt")  # FIX: audio (singular)
    print(f"\n[{tag}] input keys:", {k: tuple(v.shape) for k, v in inputs.items()})
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    pred = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
    print(f"[{tag}] pred: {pred[:120]!r}")
    return pred

preds = []
for ex in val:
    wav, sr = read_wav(ROOT/ex["wav"])
    preds.append(run(wav, sr, ex["id"]))
# zeroed-audio control (same length as clip 0)
wav0, sr0 = read_wav(ROOT/val[0]["wav"])
preds.append(run(np.zeros_like(wav0), sr0, "ZERO-CONTROL"))

print("\n=== VERDICT ===")
print("unique preds among 3 real clips:", len(set(preds[:3])))
print("real-clip-0 == zero-control:", preds[0] == preds[3])
print("If unique==1 and real==zero → audio is NOT reaching the model.")
