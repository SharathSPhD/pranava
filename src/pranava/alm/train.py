"""Phase 2 — train the Sphoṭa Projector (speech understanding).

Stage-1 audio-LLM recipe: FROZEN Parakeet encoder + FROZEN prabhasa core; train ONLY the projector
on teacher-forced byte cross-entropy (audio tokens as prefix → predict the target text bytes).
Encoder features are frozen, so we cache them once and train fast on the cache.

Eval: greedy character error rate (CER) on held-out val, audio-conditioned vs a no-audio baseline
(the core decoding from no audio prefix). If audio CER << no-audio CER, the projector conveys speech
content through the sphoṭa workspace. Runs in prabhasa/nemo-gb10.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.data import CORPUS_DIR, load_manifest, read_wav, text_to_bytes
from pranava.alm.projector import SphotaProjector

FEAT_DIR = CORPUS_DIR / "feats"


def _levenshtein(a: list[int], b: list[int]) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def extract_features(encoder_model: str = "nvidia/parakeet-tdt-0.6b-v3") -> int:
    """Cache frozen Parakeet frames for every clip → feats/{id}.npy. Returns encoder dim."""
    from pranava.alm.encoder import ParakeetEncoder

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    enc = ParakeetEncoder(model_name=encoder_model).load()
    d = 0
    for ex in load_manifest():
        out = FEAT_DIR / f"{ex.id}.npy"
        if out.exists():
            continue
        wav, sr = read_wav(ex.wav_path)
        frames = enc.encode(wav, sr=sr)[0].to("cpu").numpy().astype(np.float16)
        d = frames.shape[-1]
        np.save(out, frames)
    if d == 0:  # all cached; read one to get dim
        any_ex = load_manifest()[0]
        d = np.load(FEAT_DIR / f"{any_ex.id}.npy").shape[-1]
    return d


def _load_feat(ex, device) -> torch.Tensor:
    arr = np.load(FEAT_DIR / f"{ex.id}.npy").astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, T, D)


@torch.no_grad()
def _greedy_cer(core: SanskritCore, proj, ex_list, bias, use_audio: bool) -> float:
    dev = core.torch_device
    total_ed = total_len = 0
    for ex in ex_list:
        tgt = text_to_bytes(ex.text)
        if use_audio:
            feats = _load_feat(ex, dev)
            prefix = proj(feats, structural_bias=bias)
        else:
            # no-audio baseline: a single structural-bias "bos" embedding
            prefix = bias.view(1, 1, -1)
        out = core.greedy_from_embeds(prefix, max_new=len(tgt) + 4)
        # trim at first NUL and to target length window for a fair CER
        out = [b for b in out if b != 0][: len(tgt)]
        total_ed += _levenshtein(out, tgt)
        total_len += max(1, len(tgt))
    return total_ed / total_len


def train(arm: str = "m2", epochs: int = 8, lr: float = 1e-3, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    core = SanskritCore(arm=arm, device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev = core.torch_device
    bias = core.structural_bias

    d_enc = int(np.load(FEAT_DIR / f"{load_manifest()[0].id}.npy").shape[-1])
    proj = SphotaProjector(d_enc=d_enc, d_model=core.d_model, downsample=4).to(dev)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()

    train_ex = load_manifest("train")
    val_ex = load_manifest("val")

    def loss_on(ex) -> torch.Tensor:
        feats = _load_feat(ex, dev)
        ids = torch.tensor([text_to_bytes(ex.text)], dtype=torch.long, device=dev)
        audio_tok = proj(feats, structural_bias=bias)
        tgt_emb = core.embed_tokens(ids[:, :-1])
        x = torch.cat([audio_tok, tgt_emb], dim=1)
        for b in core.model.blocks:
            x = b(x)
        logits = core.model.head(core.model.norm_f(x))
        n = ids.shape[1]
        return ce(logits[:, -n:, :].reshape(-1, core.vocab_size), ids.reshape(-1))

    history = []
    for ep in range(epochs):
        proj.train()
        order = torch.randperm(len(train_ex))
        running = 0.0
        for idx in order:
            ex = train_ex[int(idx)]
            opt.zero_grad()
            L = loss_on(ex)
            L.backward()
            opt.step()
            running += float(L.item())
        history.append(round(running / len(train_ex), 4))

    proj.eval()
    cer_audio = _greedy_cer(core, proj, val_ex, bias, use_audio=True)
    cer_noaudio = _greedy_cer(core, proj, val_ex, bias, use_audio=False)

    ckpt = CORPUS_DIR.parent / "projector.pt"
    torch.save({"state_dict": proj.state_dict(), "d_enc": d_enc, "d_model": core.d_model,
                "arm": arm}, ckpt)
    metrics = {
        "arm": arm, "epochs": epochs, "n_train": len(train_ex), "n_val": len(val_ex),
        "train_loss_history": history,
        "val_cer_audio": round(cer_audio, 4),
        "val_cer_noaudio_baseline": round(cer_noaudio, 4),
        "cer_improvement": round(cer_noaudio - cer_audio, 4),
        "beats_baseline": bool(cer_audio < cer_noaudio),
        "projector_ckpt": str(ckpt),
    }
    (CORPUS_DIR.parent / "p2_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    import sys

    print("extracting encoder features (frozen Parakeet)...")
    extract_features()
    print("training projector...")
    m = train(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    print(json.dumps(m, indent=2))
