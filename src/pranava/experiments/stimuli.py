"""Controlled stimulus set for the holism / meaning-resolution experiments (E1).

Each stimulus is a sentence with:
  * a **meaning label** (a small discrete class the sentence ultimately conveys),
  * a **resolution class**: does the meaning resolve EARLY or LATE in the utterance,
  * the **disambiguation word index** (0-based) — the word at/after which the meaning
    is determined,
  * a **structure** tag (canonical / garden_path / verb_final / verb_first).

The design operationalises Vākyapadīya 2.143 (*pratibhā* as the distinct sentence-meaning
that arises only when word-meanings are grasped together): LATE-resolving items should show
a late, concentrated emergence of decodable meaning; EARLY items a gradual one. The
speech-vs-text contrast tests whether continuous acoustic representations resolve meaning
differently from discrete text tokens.

Stimuli are template-generated with slot fills so labels and disambiguation indices are
exact by construction — no post-hoc annotation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product


@dataclass(frozen=True, slots=True)
class Stimulus:
    id: str
    text: str
    meaning_label: str
    resolution: str  # "early" | "late"
    structure: str  # "canonical" | "garden_path" | "verb_final" | "verb_first"
    disambig_word_index: int
    n_words: int
    group: str  # the template string; CV holds out whole templates → no lexical leakage

    def to_dict(self) -> dict:
        return asdict(self)


# --- Template banks -------------------------------------------------------------
# EARLY-resolving: the main predicate/meaning is fixed by the second content word.
# Meaning label = the coarse action/outcome class, decided early.
_EARLY = [
    # (template, label, disambig_index) — subject verb object; verb (idx 1) fixes it
    ("The {subj} opened the {obj}.", "open", 2),
    ("The {subj} broke the {obj}.", "break", 2),
    ("The {subj} bought the {obj}.", "acquire", 2),
    ("The {subj} cleaned the {obj}.", "clean", 2),
]
_EARLY_SUBJ = ["woman", "farmer", "child", "sailor", "teacher", "artist"]
_EARLY_OBJ = ["door", "window", "box", "bottle", "gate", "chest"]

# LATE-resolving: garden-path & verb-final items where meaning flips/settles at the end.
# The disambiguating word is the LAST content word.
_LATE_GARDEN = [
    ("The old {subj} the {obj}.", "noun_is_verb", 3),        # "The old man the boat" — 'man'=verb
    ("The complex {subj} the {obj}.", "noun_is_verb", 3),
]
_LATE_GARDEN_SUBJ = ["man", "sailors", "police", "guards", "women", "workers"]
_LATE_GARDEN_OBJ = ["boat", "station", "houses", "post", "docks", "camp"]

_LATE_VERBFINAL = [
    ("After the storm the {subj} the harbor {verb}.", "vf", 5),
    ("Before the meeting the {subj} the report {verb}.", "vf", 5),
]
_LATE_VF_SUBJ = ["captain", "manager", "officer", "engineer", "warden", "chief"]
_LATE_VF_VERB = ["abandoned", "inspected", "praised", "rejected"]

# VERB-FIRST vs canonical (verb-first ontology probe): imperative/VS order resolves action early.
_VERB_FIRST = [
    ("Open the {obj} slowly now.", "open", 0),
    ("Break the {obj} carefully please.", "break", 0),
    ("Bring the {obj} over here.", "bring", 0),
    ("Carry the {obj} across quickly.", "carry", 0),
]
_VF_OBJ = ["door", "window", "box", "bottle", "gate", "chest"]


def _fill(template: str, **slots: str) -> str:
    return template.format(**slots)


def generate_stimuli() -> list[Stimulus]:
    out: list[Stimulus] = []
    n = 0

    def add(text: str, label: str, res: str, struct: str, dis: int, group: str) -> None:
        nonlocal n
        words = text.rstrip(".").split()
        out.append(
            Stimulus(
                id=f"s{n:04d}",
                text=text,
                meaning_label=label,
                resolution=res,
                structure=struct,
                disambig_word_index=dis,
                n_words=len(words),
                group=group,
            )
        )
        n += 1

    for (tmpl, label, dis), subj, obj in product(_EARLY, _EARLY_SUBJ, _EARLY_OBJ):
        add(_fill(tmpl, subj=subj, obj=obj), label, "early", "canonical", dis, tmpl)

    for (tmpl, label, dis), subj, obj in product(_LATE_GARDEN, _LATE_GARDEN_SUBJ, _LATE_GARDEN_OBJ):
        add(_fill(tmpl, subj=subj, obj=obj), label, "late", "garden_path", dis, tmpl)

    for (tmpl, label, dis), subj, verb in product(_LATE_VERBFINAL, _LATE_VF_SUBJ, _LATE_VF_VERB):
        add(_fill(tmpl, subj=subj, verb=verb), verb, "late", "verb_final", dis, tmpl)

    for (tmpl, label, dis), obj in product(_VERB_FIRST, _VF_OBJ):
        add(_fill(tmpl, obj=obj), label, "early", "verb_first", dis, tmpl)

    return out
