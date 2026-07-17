"""Scaled verb-final stimulus pool for E6 (tightening the key holism CI).

Verb-final sentences: the meaning label = the sentence-final verb, so meaning genuinely
resolves at the last word. A larger, more varied pool than E1's 48 verb_final items, kept
SEPARATE from the fixed 288-item E1 set so E1/E2 stay reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product


@dataclass(frozen=True, slots=True)
class VFStimulus:
    id: str
    text: str
    meaning_label: str  # the final verb
    disambig_word_index: int
    n_words: int
    group: str

    def to_dict(self) -> dict:
        return asdict(self)


# 8 templates, all ending in the final verb slot; the verb is the meaning.
_TEMPLATES = [
    "After the storm the {agent} the {object} {verb}.",
    "Before the meeting the {agent} the {object} {verb}.",
    "During the inspection the {agent} the {object} {verb}.",
    "Once the alarm stopped the {agent} the {object} {verb}.",
    "While the crowd waited the {agent} the {object} {verb}.",
    "After the long delay the {agent} the {object} {verb}.",
    "Because of the order the {agent} the {object} {verb}.",
    "When the shift ended the {agent} the {object} {verb}.",
]
_AGENTS = ["captain", "manager", "officer", "engineer", "warden", "inspector"]
_OBJECTS = ["report", "harbor", "vessel", "cargo", "bridge", "record"]
_VERBS = ["abandoned", "inspected", "praised", "rejected", "repaired", "seized", "buried", "mapped"]


def generate_verbfinal_scaled(max_items: int = 240) -> list[VFStimulus]:
    """Balanced across the 8 verb classes; label = final verb; disambig index = last word."""
    out: list[VFStimulus] = []
    n = 0
    # order agent/object outer, template innermost → consecutive combos span all templates,
    # so each verb's slice covers many groups (needed for template-grouped CV).
    combos = [(t, a, o) for a, o, t in product(_AGENTS, _OBJECTS, _TEMPLATES)]
    for verb in _VERBS:
        per_verb = max_items // len(_VERBS)
        for tmpl, agent, obj in combos[:per_verb]:
            text = tmpl.format(agent=agent, object=obj, verb=verb)
            words = text.rstrip(".").split()
            out.append(
                VFStimulus(
                    id=f"vf{n:04d}",
                    text=text,
                    meaning_label=verb,
                    disambig_word_index=len(words) - 1,
                    n_words=len(words),
                    group=tmpl,
                )
            )
            n += 1
    return out
