"""Host-side rescore: recompute cer_norm/cer_raw for every model from the SAVED raw per-clip predictions.

The GPU run stores raw predictions per model; scoring is cheap and deterministic, so we finalize the
metric here with the best normalizer (SLP1→IAST and Devanagari→IAST via indic-transliteration, then an
ASCII phonetic fold). Run with the pranava venv: .venv/bin/python scripts/alm/_rescore_alm.py
"""
import json
import re
import unicodedata
from pathlib import Path

from indic_transliteration import sanscript

_TAG = re.compile(r"^\s*\[[a-zA-Z]{2,4}\]\s*")  # leading language/format tag e.g. "[sa] " — strip uniformly
_CONT = re.compile(r"(?is)\b(human|assistant|user|system)\s*:.*$")  # drop chat-style continuations


def _first_answer(text: str) -> str:
    """Take the transcription only: first non-empty line, minus any chat continuation. Uniform cleanup
    so a model that transcribes correctly then rambles (e.g. Qwen2.5-Omni appends 'Human: ...') is scored
    on its transcription, not its verbosity."""
    text = _CONT.sub("", text)
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text.strip()

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "benchmark"


def _fold(s: str) -> str:
    """NFKD → strip combining → lowercase → keep ONLY ascii [a-z0-9 ] → collapse ws.

    ASCII-only keep drops any residual non-Latin script (untransliterated Devanagari/CJK) so all models
    are compared on the same romanized phonetic skeleton."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = "".join(c if (("a" <= c <= "z") or ("0" <= c <= "9") or c.isspace()) else " " for c in s)
    return " ".join(s.split())


def _has_deva(s: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in s)


def norm(text: str, is_slp1: bool) -> str:
    if not is_slp1:
        text = _first_answer(text)  # generalists: score the transcription line, not trailing chatter
    text = _TAG.sub("", text)  # drop a leading language/format tag uniformly for all models
    if is_slp1:
        text = sanscript.transliterate(text, sanscript.SLP1, sanscript.IAST)
    elif _has_deva(text):
        try:
            text = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
        except Exception:
            pass
    return _fold(text)


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


def cer(p, g):
    return _lev(p, g) / max(1, len(g))


def is_slp1_model(name: str) -> bool:
    return "Śabda" in name or "ours" in name


def main() -> int:
    lb = json.loads((OUT / "alm_vs_alm.json").read_text())
    recs = json.loads((OUT / "alm_vs_alm_records.json").read_text())

    for entry in lb["leaderboard"]:
        name = entry["model"]
        per = recs.get(name, [])
        if not per:
            continue
        slp1 = is_slp1_model(name)
        raws, norms, uniq = [], [], set()
        for r in per:
            p, g = r["pred"], r["gold"]
            uniq.add(p)
            crraw = cer(p, g)                        # raw: pred as-is vs SLP1 gold (specialist-native scheme)
            cn = cer(norm(p, slp1), norm(g, True))   # normalized: scheme-neutral phonetic skeleton
            r["cer_raw"], r["cer_norm"] = round(crraw, 4), round(cn, 4)
            raws.append(crraw); norms.append(cn)
        entry["cer_raw"] = round(sum(raws) / len(raws), 4)
        entry["cer_norm"] = round(sum(norms) / len(norms), 4)
        entry["n_scored"] = len(norms)
        entry["unique_outputs"] = len(uniq)
        if entry.get("alm_type", "").startswith("generalist"):
            entry["audio_conditioned"] = len(uniq) > 1

    lb["leaderboard"].sort(key=lambda r: (r["cer_norm"] is None, r["cer_norm"] if r["cer_norm"] is not None else 9))
    lb["rescored"] = "host rescore: SLP1/Devanagari→IAST + ASCII phonetic fold (indic-transliteration)"
    (OUT / "alm_vs_alm.json").write_text(json.dumps(lb, indent=2, ensure_ascii=False))
    (OUT / "alm_vs_alm_records.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False))

    print(f"{'model':52} {'cer_norm':>9} {'cer_raw':>9} {'uniq':>5} {'n':>4}")
    for e in lb["leaderboard"]:
        print(f"{e['model'][:52]:52} {str(e.get('cer_norm')):>9} {str(e.get('cer_raw')):>9} "
              f"{str(e.get('unique_outputs')):>5} {str(e.get('n_scored')):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
