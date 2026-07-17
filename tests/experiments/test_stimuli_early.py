"""TDD: matched early pool — same verbs as verb-final, verb resolves early."""
from collections import Counter

from pranava.experiments.stimuli_early import generate_early_scaled
from pranava.experiments.stimuli_verbfinal import generate_verbfinal_scaled


def test_size_and_balance():
    s = generate_early_scaled()
    assert len(s) >= 200
    c = Counter(x.meaning_label for x in s)
    assert len(c) == 8 and max(c.values()) - min(c.values()) <= 1


def test_verb_is_early():
    for x in generate_early_scaled():
        assert x.disambig_word_index <= 2
        assert x.text.split()[2].rstrip(".") == x.meaning_label


def test_matched_vocabulary_with_verbfinal():
    early_verbs = {x.meaning_label for x in generate_early_scaled()}
    late_verbs = {x.meaning_label for x in generate_verbfinal_scaled()}
    assert early_verbs == late_verbs  # identical label space → clean comparison


def test_templates_for_grouped_cv():
    assert len({x.group for x in generate_early_scaled()}) >= 4
