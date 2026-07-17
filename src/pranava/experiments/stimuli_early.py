"""Matched early-resolving pool for E7 — same 8 verbs as the verb-final pool, but early position.

Holds vocabulary constant with stimuli_verbfinal so the ONLY manipulation is *when* the meaning
(the verb) resolves. Early = verb at ~index 2; late = verb at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product

# same verb classes as the verb-final pool, for a matched comparison
_VERBS = ["abandoned", "inspected", "praised", "rejected", "repaired", "seized", "buried", "mapped"]
_AGENTS = ["captain", "manager", "officer", "engineer", "warden", "inspector"]
_OBJECTS = ["report", "harbor", "vessel", "cargo", "bridge", "record"]
# 8 trailing adjuncts → 8 templates, verb stays at index 2 (resolves early)
_ADJUNCTS = [
    "after the storm", "before the meeting", "during the inspection", "once the alarm stopped",
    "while the crowd waited", "after the long delay", "because of the order", "when the shift ended",
]


@dataclass(frozen=True, slots=True)
class EarlyStimulus:
    id: str
    text: str
    meaning_label: str
    disambig_word_index: int
    n_words: int
    group: str

    def to_dict(self) -> dict:
        return asdict(self)


def generate_early_scaled(max_items: int = 240) -> list[EarlyStimulus]:
    out: list[EarlyStimulus] = []
    n = 0
    # adjunct (the "template") innermost so each verb's slice spans all 8 groups
    combos = [(adj, a, o) for a, o, adj in product(_AGENTS, _OBJECTS, _ADJUNCTS)]
    for verb in _VERBS:
        for adj, agent, obj in combos[: max_items // len(_VERBS)]:
            text = f"The {agent} {verb} the {obj} {adj}."
            words = text.rstrip(".").split()
            out.append(
                EarlyStimulus(
                    id=f"ea{n:04d}",
                    text=text,
                    meaning_label=verb,
                    disambig_word_index=2,  # "The <agent> <verb> ..." → verb at index 2
                    n_words=len(words),
                    group=adj,
                )
            )
            n += 1
    return out
