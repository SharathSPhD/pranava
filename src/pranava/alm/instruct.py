"""Instruction-following for the Śabda-ALM — one audio input, many tasks.

The core is a byte-level decoder that already accepts an embedding *prefix* (the projected audio
tokens). We make it instruction-conditioned by embedding a short English **instruction** through the
core's own byte-embedding and concatenating it *after* the audio tokens:

    [audio tokens (madhyamā)] + [instruction bytes] → generate [answer bytes]

Every answer is derived from gold annotations already on disk — the kāraka parse (kartā / karaṇa /
karma / kriyā) and the language tag — so no target is fabricated. The instruction merely *selects*
which structured fact the model must read out of the same audio. This is the SFT substrate; RLAIF
then does a preference pass on top (see scripts/alm/train_rlaif.py).

The template/extraction functions here are pure (host-testable). ``build_prefix`` needs the core and
runs on GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

# Natural short instructions (English) → the gold role they read out. Kept short so the instruction
# byte-prefix is cheap. The model must learn the *mapping* instruction→field, not memorise one task.
INSTRUCTIONS: dict[str, str] = {
    "transcribe": "transcribe the speech",
    "kriya": "what is the action",
    "karta": "who is the agent",
    "karma": "what is the object",
    "karana": "by what means",
    "language": "which language is this",
}

# gold kāraka role name (prabhasa tagset) for each extractive task
_ROLE = {"kriya": "kriyA", "karta": "karwA", "karma": "karma", "karana": "karaNam"}

# End-of-response sentinel. Byte 0 (NUL) never occurs in the ASCII responses, so the model can learn
# it as an unambiguous "I'm done" signal — without it, greedy decoding runs to max_new and rambles.
EOS = 0


@dataclass(frozen=True, slots=True)
class InstructionExample:
    wav: str
    instruction: str  # the English instruction text fed after the audio
    response: str      # the gold answer bytes the model must produce
    task: str
    lang: str
    split: str


def _karaka_map(karaka: list) -> dict[str, str]:
    """[[filler, role], …] → {role: filler}. Ignores malformed pairs."""
    return {role: filler for pair in karaka if len(pair) == 2 for filler, role in [pair]}


def tasks_for(*, wav: str, text: str, karaka: list, lang: str, split: str) -> list[InstructionExample]:
    """All instruction examples a single clip supports, using only gold labels present on it.

    Sanskrit clips (with a kāraka parse) yield transcribe + language + every kāraka role that is
    actually filled. English clips (no parse) yield transcribe + language. A task is emitted only
    when its gold answer exists — never invented.
    """
    clean = text.split("]", 1)[1].strip() if text[:5] in ("[sa] ", "[en] ") else text.strip()
    lang_word = {"sa": "Sanskrit", "en": "English"}.get(lang, lang)
    out = [
        InstructionExample(wav, INSTRUCTIONS["transcribe"], clean, "transcribe", lang, split),
        InstructionExample(wav, INSTRUCTIONS["language"], lang_word, "language", lang, split),
    ]
    kmap = _karaka_map(karaka)
    for task, role in _ROLE.items():
        if role in kmap:
            out.append(InstructionExample(wav, INSTRUCTIONS[task], kmap[role], task, lang, split))
    return out


def build_prefix(core, audio_tokens, instruction: str):
    """[audio tokens] ++ [instruction-byte embeddings] → a decoder prefix (1, T, d).

    ``audio_tokens`` (1, Ta, d) already live in the core's structural-bias space (the projector
    outputs there); the instruction bytes are embedded via ``core.embed_tokens`` which adds the same
    bias, so the two halves are commensurate.
    """
    import torch

    ids = torch.tensor([list(instruction.encode("utf-8"))], dtype=torch.long,
                       device=core.torch_device)
    instr_embeds = core.embed_tokens(ids)
    return torch.cat([audio_tokens.to(core.torch_device), instr_embeds], dim=1)
