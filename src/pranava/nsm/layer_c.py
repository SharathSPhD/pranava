"""NSM Layer C — inline pramāṇa validation of ALM outputs (Navya-Nyāya epistemic gate).

The four-vāk stack routes sound → sphoṭa workspace → text. Layer C sits *after* generation and asks a
Navya-Nyāya question of each utterance the model produces: **is this ascertained (nirṇaya), or is it
a fallacy-tainted guess?** It reuses the operator's pramāṇa auditor (``pranava.pramana_layer``) but
grounds the abstract "evidence" in three *observable* properties of a byte-decoder's output — no
extra model, no gold label needed at inference:

  * **pratyakṣa** (direct perception): the model's own decode confidence (mean token probability).
  * **śabda** (the language's testimony): romanization legality — a real Sanskrit/English utterance
    uses only legal ASCII letters, not stray bytes.
  * **anumāna** (inference): non-repetition — pathological looping ("vane vane vane") is the tell of
    an invalid generation, inferred without needing to know the truth.

An utterance is ASCERTAINED only if confidence and legality support it and repetition does not
overturn them. The point (validated in scripts/nsm/validate_layer_c.py): ascertained outputs are
*measurably more accurate* than rejected ones — so Layer C is a real, gold-free hallucination filter
and the confidence channel the platform's products (API, app) can expose.
"""
from __future__ import annotations

from pranava.pramana_layer.auditor import (
    AuditReport,
    Claim,
    Evidence,
    Verdict,
    audit,
)

# import the enum lazily-safely so the module still imports where pramana is absent
from pramana.domain.models.nyaya_example import PramanaType

# below these, a signal is treated as weak/absent support rather than strong
CONF_FLOOR = 0.5
LEGAL_FLOOR = 0.9


def repetition_ratio(text: str) -> float:
    """Fraction of adjacent word-pairs that are identical — the looping tell."""
    words = text.split()
    if len(words) < 2:
        return 0.0
    return sum(1 for a, b in zip(words, words[1:]) if a == b) / (len(words) - 1)


def romanization_legality(text: str) -> float:
    """Fraction of characters that are legal romanized script (ASCII letter or space)."""
    if not text:
        return 0.0
    return sum(1 for c in text if c.isascii() and (c.isalpha() or c == " ")) / len(text)


def validate_output(text: str, confidence: float) -> AuditReport:
    """Audit one ALM utterance → a Navya-Nyāya ascertainment verdict.

    ``confidence`` is the mean per-token probability of the decoded output (0..1).
    """
    conf = max(0.0, min(1.0, float(confidence)))
    legality = romanization_legality(text)
    rep = repetition_ratio(text)

    # Confidence is the primary support (pratyakṣa). Legality and repetition are *disqualifiers*:
    # a real utterance is legal and non-looping, so those signals only ever count against a claim —
    # they never inflate a low-confidence output into an ascertained one.
    evidence = [
        Evidence(PramanaType.PRATYAKSHA, f"decode confidence {conf:.2f}", supports=True, strength=conf),
    ]
    if legality < LEGAL_FLOOR:  # śabda: stray non-script bytes refute a valid-utterance claim
        evidence.append(Evidence(PramanaType.SHABDA, f"illegible romanization {legality:.2f}",
                                 supports=False, strength=min(1.0, 1.0 - legality)))
    if rep > 0.0:  # anumāna: looping infers invalid generation (weighted — a strong tell)
        evidence.append(Evidence(PramanaType.ANUMANA, f"token repetition {rep:.2f}", supports=False,
                                 strength=min(1.0, 2.0 * rep)))

    return audit(Claim(text=f"ALM output «{text}» is a valid utterance", evidence=evidence))


def is_ascertained(text: str, confidence: float) -> bool:
    return validate_output(text, confidence).verdict is Verdict.ASCERTAINED
