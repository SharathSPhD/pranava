"""Build the instruction-tuning corpus from gold-annotated speech.

Reshapes the native-Sanskrit corpus (speech_corpus_indic — has the gold kāraka parse) and the real
English corpus (speech_corpus_en — LibriSpeech) into (wav, instruction, response) tuples across six
tasks (transcribe / language / kartā / karaṇa / karma / kriyā). Every response is a gold label
already on disk; nothing is invented. Host-only (no GPU): it just reindexes existing manifests.

    python scripts/alm/build_instruction_data.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _root() -> Path:
    for c in ("/work/pranava", "/home/sharaths/projects/pranava"):
        if (Path(c) / "data/alm").exists():
            return Path(c)
    return Path(__file__).resolve().parents[2]


def main() -> None:
    import sys

    root = _root()
    sys.path.insert(0, str(root / "src"))
    from pranava.alm.instruct import tasks_for

    sources = [("speech_corpus_indic", "sa"), ("speech_corpus_en", "en")]
    examples = []
    for corpus, lang in sources:
        man = root / "data/alm" / corpus / "manifest.jsonl"
        if not man.exists():
            print(f"skip {corpus} (no manifest)")
            continue
        for line in man.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            wav = str(root / r["wav"]) if not str(r["wav"]).startswith("/") else r["wav"]
            examples.extend(_as_dicts(
                tasks_for(wav=wav, text=r["text"], karaka=r.get("karaka", []),
                          lang=r.get("lang") or lang, split=r["split"])
            ))

    out_dir = root / "data/alm/instruct_corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    by_task = Counter(e["task"] for e in examples)
    by_lang = Counter(e["lang"] for e in examples)
    by_split = Counter(e["split"] for e in examples)
    datasheet = {
        "n_examples": len(examples),
        "by_task": dict(by_task),
        "by_language": dict(by_lang),
        "by_split": dict(by_split),
        "sources": "speech_corpus_indic (native Sanskrit + gold kāraka) + speech_corpus_en (LibriSpeech)",
        "note": "every response is a gold label (kāraka filler / transcript / language) already on "
                "disk — instructions select which field to read; nothing is fabricated.",
    }
    (out_dir / "datasheet.json").write_text(json.dumps(datasheet, indent=2, ensure_ascii=False))
    print(json.dumps(datasheet, indent=2, ensure_ascii=False))


def _as_dicts(examples):
    return [{"wav": e.wav, "instruction": e.instruction, "response": e.response,
             "task": e.task, "lang": e.lang, "split": e.split} for e in examples]


if __name__ == "__main__":
    main()
