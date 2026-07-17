"""Pramāṇa validation layer (NSM "Layer C") — a Navya-Nyāya epistemic auditor.

Reuses the operator's published *pramana* project (arXiv 2604.04937): its 6-phase Nyāya
methodology and the ``HetvabhasaType`` fallacy taxonomy (imported via a venv .pth). Here we apply
that framework not to puzzle-solving but to **pranava's own empirical claims** — a self-referential
epistemic audit. A claim is *ascertained* (nirṇaya) only if it clears every phase; a fallacious
reason (hetvābhāsa) blocks ascertainment.

The design point: this is the epistemic gate the docs call for — before a claim about speech-vs-text
cognition is asserted, it must survive śabda/anumāna evidence, counterfactual tarka, and fallacy
detection. Applied to the E2 holism claim, it independently flags the underpowered-CV reason as
*savyabhichāra* (an inconclusive/erratic hetu) — the same defect E6/E7 found empirically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pramana.domain.models.nyaya_example import HetvabhasaType, PramanaType


class Verdict(str, Enum):
    ASCERTAINED = "ascertained"        # nirṇaya: claim clears all phases
    NOT_ASCERTAINED = "not_ascertained"  # blocked by a fallacy (hetvābhāsa)
    DOUBTFUL = "doubtful"              # insufficient pramāṇa; saṃśaya unresolved


@dataclass(frozen=True, slots=True)
class Evidence:
    """A single evidential item backing (or undermining) a claim."""
    pramana_type: PramanaType  # how it is known: pratyakṣa / anumāna / upamāna / śabda
    statement: str
    supports: bool = True
    strength: float = 1.0  # 0..1; e.g. an underpowered test has low strength


@dataclass(frozen=True, slots=True)
class Fallacy:
    kind: HetvabhasaType
    explanation: str


@dataclass(slots=True)
class Claim:
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    declared_fallacies: list[Fallacy] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AuditReport:
    claim: str
    doubt_type: str
    pramana_used: list[str]
    hetu: str
    fallacies: list[Fallacy]
    verdict: Verdict
    nirnaya: str  # the concluding statement

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "samshaya": self.doubt_type,
            "pramana_used": self.pramana_used,
            "hetu": self.hetu,
            "hetvabhasa": [{"kind": f.kind.value, "explanation": f.explanation} for f in self.fallacies],
            "verdict": self.verdict.value,
            "nirnaya": self.nirnaya,
        }


# thresholds for the epistemic check
_MIN_SUPPORT_STRENGTH = 0.5
_MIN_NET_SUPPORT = 0.0


def audit(claim: Claim) -> AuditReport:
    """Run a claim through the 6-phase Nyāya check and return an ascertainment verdict.

    Phases: saṃśaya (doubt) → pramāṇa (evidence sources) → pañca-avayava (the hetu) →
    tarka (net-support counterfactual) → hetvābhāsa (fallacy detection) → nirṇaya.
    """
    supporting = [e for e in claim.evidence if e.supports]
    opposing = [e for e in claim.evidence if not e.supports]
    pramana_used = sorted({e.pramana_type.value for e in claim.evidence})

    # Saṃśaya: a doubt exists when there is any opposing evidence or weak support.
    has_doubt = bool(opposing) or any(e.strength < _MIN_SUPPORT_STRENGTH for e in supporting)
    doubt_type = "conflicting-or-weak-evidence" if has_doubt else "prima-facie-clear"

    # Pañca-avayava hetu: the strongest supporting reason (or "none").
    if supporting:
        hetu = max(supporting, key=lambda e: e.strength).statement
    else:
        hetu = "no supporting reason offered"

    # Tarka: net support = Σ strengths(support) − Σ strengths(oppose).
    net = sum(e.strength for e in supporting) - sum(e.strength for e in opposing)

    # Hetvābhāsa detection.
    fallacies: list[Fallacy] = list(claim.declared_fallacies)
    if supporting and all(e.strength < _MIN_SUPPORT_STRENGTH for e in supporting):
        fallacies.append(Fallacy(HetvabhasaType.SAVYABHICHARA,
                                 "the reason does not invariably yield the conclusion "
                                 "(weak/underpowered support)"))
    if opposing and sum(e.strength for e in opposing) >= sum(e.strength for e in supporting):
        fallacies.append(Fallacy(HetvabhasaType.SATPRATIPAKSHA,
                                 "an equally strong counter-reason exists"))
    if not claim.evidence:
        fallacies.append(Fallacy(HetvabhasaType.ASIDDHA, "the reason itself is unestablished"))

    # Nirṇaya.
    if fallacies:
        verdict = Verdict.NOT_ASCERTAINED
        nirnaya = f"Not ascertained: blocked by {', '.join(f.kind.value for f in fallacies)}."
    elif not supporting or net <= _MIN_NET_SUPPORT:
        verdict = Verdict.DOUBTFUL
        nirnaya = "Doubtful: insufficient net pramāṇa to ascertain."
    else:
        verdict = Verdict.ASCERTAINED
        nirnaya = "Ascertained: clears saṃśaya, pramāṇa, tarka, and hetvābhāsa checks."

    return AuditReport(claim.text, doubt_type, pramana_used, hetu, fallacies, verdict, nirnaya)
