"""Fair free-decode eval of the INSTRUCT specialist (EOS-trained) on the 58-clip ALM benchmark.

The 1.82 free-decode CER came from lora_ckpt, which was never trained to stop. The instruct
checkpoint has an EOS sentinel (byte 0) and stop-at-EOS decoding — its natural, oracle-free decode.
Same protocol as the generalists: greedy, fixed 64-byte budget, stop at the model's own EOS, no gold
information. Appends per-clip predictions into alm_vs_alm_records.json under a new model row.

Run: scripts/alm/in_container.sh python /work/pranava/scripts/alm/bench_specialist_eos.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "benchmark"
CORPUS = ROOT / "data" / "alm" / "speech_corpus_indic"
FEAT_DIR = CORPUS / "feats"
MODEL_NAME = "Śabda-ALM instruct+EOS — free decode (ours)"


def main() -> int:
    from pranava.alm.core_adapter import SanskritCore
    from pranava.alm.encoder import ParakeetEncoder
    from pranava.alm.instruct import EOS, build_prefix
    from pranava.alm.lora import inject_lora
    from pranava.alm.projector import SphotaProjector

    rows = [json.loads(x) for x in (CORPUS / "manifest.jsonl").open(encoding="utf-8") if x.strip()]
    val = [r for r in rows if r["split"] == "val"][:58]

    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    inject_lora(core.model, r=8, alpha=16)
    blob = torch.load(ROOT / "data/alm/instruct_ckpt.pt", map_location=core.torch_device, weights_only=True)
    sd = {n: p for n, p in core.model.named_parameters() if ".A" in n or ".B" in n}
    for n, w in blob["lora"].items():
        if n in sd:
            sd[n].data.copy_(w.to(core.torch_device))
    proj = SphotaProjector(d_enc=blob["d_enc"], d_model=blob["d_model"], downsample=4).to(core.torch_device)
    proj.load_state_dict(blob["projector"]); proj.eval()
    bias = core.structural_bias
    enc = None

    per_clip = []
    with torch.no_grad():
        for r in val:
            fp = FEAT_DIR / f"{r['id']}.npy"
            if fp.exists():
                feats = np.load(fp).astype(np.float32)
                t = torch.from_numpy(feats).unsqueeze(0).to(core.torch_device)
            else:
                if enc is None:
                    enc = ParakeetEncoder().load()
                from pranava.alm.data import read_wav
                wav, sr = read_wav(ROOT / r["wav"])
                t = enc.encode(wav, sr=sr)
                if not torch.is_tensor(t):
                    t = torch.from_numpy(np.asarray(t))
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                t = t.to(core.torch_device)
            tokens = proj(t, structural_bias=bias)
            prefix = build_prefix(core, tokens, "transcribe the speech")
            out = core.greedy_from_embeds(prefix, max_new=64, stop_token=EOS)  # model's OWN stop — fair
            pred = bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()
            per_clip.append({"id": r["id"], "pred": pred, "gold": r["text"]})
            print(f"  {r['id']}: pred={pred[:60]!r} gold={r['text'][:40]!r}", flush=True)

    recs = json.loads((OUT / "alm_vs_alm_records.json").read_text())
    recs[MODEL_NAME] = per_clip
    (OUT / "alm_vs_alm_records.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False))

    lb = json.loads((OUT / "alm_vs_alm.json").read_text())
    if not any(e["model"] == MODEL_NAME for e in lb["leaderboard"]):
        lb["leaderboard"].append({
            "model": MODEL_NAME, "params": "200M core + 0.6B enc",
            "alm_type": "specialist (Sanskrit fine-tuned, EOS-trained instruct decode)",
            "cer_norm": None, "cer_raw": None, "n_scored": len(per_clip), "unique_outputs": None,
            "note": "free decode, stop at the model's own EOS sentinel — no gold-length oracle",
        })
    (OUT / "alm_vs_alm.json").write_text(json.dumps(lb, indent=2, ensure_ascii=False))
    print(f"\nAppended {len(per_clip)} per-clip preds for: {MODEL_NAME}")
    print("Now run scripts/alm/_rescore_alm.py on the host to score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
