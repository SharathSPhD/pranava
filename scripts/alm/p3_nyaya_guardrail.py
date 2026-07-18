"""P3-N-G: Nyāya guardrail at decode time — śabda-pramāṇa self-consistency.

Pre-registered experiment to test whether a legality constraint (the model's kāraka answer must be
grounded in its OWN heard transcription, not gold access) improves kāraka accuracy.

Method:
  - For each val clip and task ∈ {karta, karma, kriya, karana}:
    1. Decode transcription (free, EOS)
    2. Decode kāraka answer (free, EOS)
    3. Legality check: answer must match a word of the transcription (edit distance ≤ 1)
    4. If illegal → constrained re-decode: byte-trie over transcription words masks illegal
       continuations (greedy argmax restricted to trie-legal next bytes, EOS legal only at word ends)
  - Report exact-match with/without guardrail vs gold
  - Fire/fix/break/neutral counts
  - Zero gold leakage by construction (constraint set derives from model's own transcript)

Pre-registered hypotheses:
  - H-NG1: guardrailed kāraka exact-match ≥ unguarded (strictly greater to claim benefit)
  - H-NG2: fire-rate and fix/break/neutral counts reported; a guardrail that fires often but fixes
    nothing is reported as such

Output: data/alm/p3ng_results.json with per-clip records and aggregate metrics.

Falsifiable: if constrained re-decode reduces accuracy (transcript itself too noisy), that is the
honest result and bounds the approach by transcription quality.

Run in container: scripts/alm/in_container.sh python /work/pranava/scripts/alm/p3_nyaya_guardrail.py
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from pranava.alm.core_adapter import SanskritCore
from pranava.alm.data import CORPUS_DIR, load_manifest, text_to_bytes
from pranava.alm.instruct import EOS, build_prefix, INSTRUCTIONS
from pranava.alm.projector import SphotaProjector

ROOT = Path("/work/pranava") if Path("/work/pranava").exists() else Path(__file__).resolve().parents[2]
FEAT_DIR = CORPUS_DIR / "feats"
OUT = ROOT / "data/alm"


@dataclass(frozen=True, slots=True)
class KarakaTaskResult:
    """Per-clip, per-task kāraka result."""
    clip_id: str
    task: str  # "karta", "karma", "kriya", "karana"
    gold_answer: str
    unguarded_answer: str
    unguarded_exact_match: bool
    guardrail_fired: bool
    guarded_answer: str
    guarded_exact_match: bool
    outcome: str  # "fix", "break", "neutral", "no_change"
    transcript: str


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance."""
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


def _is_legal(answer: str, transcript: str) -> bool:
    """Answer is legal if it matches a transcript word within edit distance 1."""
    words = transcript.split()
    for word in words:
        if _edit_distance(answer, word) <= 1:
            return True
    return False


class ByteTrie:
    """Byte-level trie for constraining next-byte predictions."""
    def __init__(self):
        self.root: dict = {}

    def add_word(self, word: str):
        """Add a word (as bytes) to the trie."""
        current = self.root
        for byte_val in word.encode("utf-8"):
            if byte_val not in current:
                current[byte_val] = {}
            current = current[byte_val]
        current["<END>"] = True

    def get_legal_bytes(self, prefix: list[int]) -> set[int]:
        """Get legal next bytes given a prefix, including EOS if at word boundary."""
        current = self.root
        for byte_val in prefix:
            if byte_val not in current:
                return set()  # no legal continuations
            current = current[byte_val]

        legal = set(current.keys())
        # Remove the <END> marker but allow it to mean "we can output EOS"
        legal.discard("<END>")

        # EOS is legal only if we're at a word boundary (the marker "<END>" was present)
        if "<END>" in current:
            legal.add(EOS)
        return legal


@torch.no_grad()
def decode_unguarded(core, prefix: torch.Tensor, instruction: str, max_new: int = 24) -> str:
    """Greedy decode with instruction, stop at EOS."""
    full_prefix = build_prefix(core, prefix, instruction)
    tokens = core.greedy_from_embeds(full_prefix, max_new=max_new, stop_token=EOS)
    return bytes(b for b in tokens if 9 <= b < 127).decode("latin-1", "ignore").strip()


@torch.no_grad()
def decode_guarded(core, prefix: torch.Tensor, instruction: str, legal_words: list[str],
                   max_new: int = 24) -> str:
    """Constrained decode restricted to legal-word byte-prefixes."""
    trie = ByteTrie()
    for word in legal_words:
        trie.add_word(word)

    full_prefix = build_prefix(core, prefix, instruction)
    x = full_prefix.to(core.torch_device)
    out: list[int] = []

    for _ in range(max_new):
        logits = core.forward_embeds(x)[0, -1]

        # Mask illegal bytes to -inf
        legal_bytes = trie.get_legal_bytes(out)
        if legal_bytes:
            mask = torch.full_like(logits, float("-inf"))
            for b in legal_bytes:
                if b < len(mask):
                    mask[b] = logits[b]
            logits = mask
        else:
            # No legal continuations; reset to base logits (allows any)
            pass

        nxt = int(torch.argmax(logits).item())
        if nxt == EOS or not torch.isfinite(logits[nxt]):
            break
        out.append(nxt)
        nxt_emb = core.model.embed(
            torch.tensor([[nxt]], device=core.torch_device, dtype=torch.long)
        )
        x = torch.cat([x, nxt_emb], dim=1)

    return bytes(b for b in out if 9 <= b < 127).decode("latin-1", "ignore").strip()


def main(n_clips: int = 40) -> int:
    core = SanskritCore(arm="m2", device="cuda").load()
    for p in core.model.parameters():
        p.requires_grad_(False)
    dev, bias = core.torch_device, core.structural_bias

    # Load checkpoint
    ckpt_paths = [ROOT / "data/alm/xl_ckpt.pt", ROOT / "data/alm/instruct_ckpt.pt"]
    blob = None
    ckpt_used = None
    for ckpt_path in ckpt_paths:
        if ckpt_path.exists():
            blob = torch.load(ckpt_path, map_location=dev, weights_only=True)
            ckpt_used = str(ckpt_path)
            break
    if blob is None:
        print(f"ERROR: no checkpoint found at {ckpt_paths}", flush=True)
        return 1

    proj = SphotaProjector(d_enc=blob.get("d_enc", 768), d_model=blob.get("d_model", 2048),
                           downsample=4).to(dev).eval()
    if "projector" in blob:
        proj.load_state_dict(blob["projector"])
    elif "state_dict" in blob:
        proj.load_state_dict(blob["state_dict"])
    print(f"Loaded checkpoint: {ckpt_used}", flush=True)

    # Load val clips with karaka annotations
    exs = [e for e in load_manifest("val") if e.karaka and (FEAT_DIR / f"{e.id}.npy").exists()][:n_clips]
    print(f"Processing {len(exs)} val clips with kāraka annotations...", flush=True)

    results: list[KarakaTaskResult] = []
    outcomes_count = {"fix": 0, "break": 0, "neutral": 0, "no_change": 0}
    fires_count = 0

    with torch.no_grad():
        for clip_idx, ex in enumerate(exs):
            if clip_idx % 10 == 0:
                print(f"  Processing clip {clip_idx}/{len(exs)}...", flush=True)

            feats = torch.from_numpy(np.load(FEAT_DIR / f"{ex.id}.npy").astype(np.float32)
                                      ).unsqueeze(0).to(dev)
            audio_prefix = proj(feats, structural_bias=bias)

            # Decode transcription (unguarded)
            transcript = decode_unguarded(core, audio_prefix, INSTRUCTIONS["transcribe"])

            # Build legal words for guardrail
            legal_words = transcript.split()

            # For each kāraka task
            tasks = ["karta", "karma", "kriya", "karana"]
            for task in tasks:
                # Find gold answer
                gold_answer = None
                role = {"karta": "karwA", "karma": "karma", "kriya": "kriyA", "karana": "karaNam"}[task]
                for pair in ex.karaka:
                    if len(pair) == 2 and pair[1] == role:
                        gold_answer = pair[0]
                        break

                if gold_answer is None:
                    continue  # skip if no gold answer

                # Decode unguarded
                unguarded = decode_unguarded(core, audio_prefix, INSTRUCTIONS[task])
                unguarded_match = unguarded == gold_answer

                # Check legality
                is_legal = _is_legal(unguarded, transcript)

                # Decode guarded if illegal
                if is_legal:
                    guarded = unguarded
                    guardrail_fired = False
                else:
                    guarded = decode_guarded(core, audio_prefix, INSTRUCTIONS[task], legal_words)
                    guardrail_fired = True
                    fires_count += 1

                guarded_match = guarded == gold_answer

                # Classify outcome
                if not guardrail_fired:
                    outcome = "no_change"
                elif guarded_match and not unguarded_match:
                    outcome = "fix"
                    outcomes_count["fix"] += 1
                elif guarded_match and unguarded_match:
                    outcome = "neutral"
                    outcomes_count["neutral"] += 1
                elif not guarded_match and unguarded_match:
                    outcome = "break"
                    outcomes_count["break"] += 1
                else:
                    outcome = "neutral"
                    outcomes_count["neutral"] += 1

                results.append(KarakaTaskResult(
                    clip_id=ex.id,
                    task=task,
                    gold_answer=gold_answer,
                    unguarded_answer=unguarded,
                    unguarded_exact_match=unguarded_match,
                    guardrail_fired=guardrail_fired,
                    guarded_answer=guarded,
                    guarded_exact_match=guarded_match,
                    outcome=outcome,
                    transcript=transcript,
                ))

    # Aggregate metrics
    if results:
        unguarded_em = np.mean([1 if r.unguarded_exact_match else 0 for r in results])
        guarded_em = np.mean([1 if r.guarded_exact_match else 0 for r in results])
        fire_rate = fires_count / len(results) if results else 0.0

        # H-NG1: guarded ≥ unguarded (strictly greater to claim benefit)
        h_ng1_satisfied = guarded_em > unguarded_em

        result = {
            "checkpoint_used": ckpt_used,
            "n_clips": len(exs),
            "n_tasks": len(results),
            "unguarded_exact_match": round(unguarded_em, 4),
            "guarded_exact_match": round(guarded_em, 4),
            "improvement": round(guarded_em - unguarded_em, 4),
            "fire_rate": round(fire_rate, 4),
            "outcomes": {k: int(v) for k, v in outcomes_count.items()},
            "hypothesis_H_NG1": {
                "claim": "guarded exact-match > unguarded",
                "unguarded_rate": round(unguarded_em, 4),
                "guarded_rate": round(guarded_em, 4),
                "satisfied": bool(h_ng1_satisfied),
            },
            "hypothesis_H_NG2": {
                "claim": "report fire-rate and fix/break/neutral counts",
                "fire_rate": round(fire_rate, 4),
                "fix_count": outcomes_count["fix"],
                "break_count": outcomes_count["break"],
                "neutral_count": outcomes_count["neutral"],
                "no_change_count": outcomes_count["no_change"],
            },
            "per_task_records": [asdict(r) for r in results[:100]],  # first 100 for inspection
        }
    else:
        result = {
            "checkpoint_used": ckpt_used,
            "error": "no valid tasks processed",
            "per_task_records": [],
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p3ng_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results written to {OUT / 'p3ng_results.json'}", flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != "per_task_records"}, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
