"""TDD: scaled verb-final pool is well-formed, balanced, and larger than E1's."""
from collections import Counter

from pranava.experiments.stimuli_verbfinal import generate_verbfinal_scaled, VFStimulus


def test_size_larger_than_e1_verbfinal():
    s = generate_verbfinal_scaled()
    assert len(s) >= 200  # E1 had 48


def test_unique_ids():
    s = generate_verbfinal_scaled()
    assert len({x.id for x in s}) == len(s)


def test_label_is_final_word():
    for x in generate_verbfinal_scaled():
        assert x.text.rstrip(".").split()[-1] == x.meaning_label
        assert x.disambig_word_index == x.n_words - 1


def test_verb_classes_balanced():
    c = Counter(x.meaning_label for x in generate_verbfinal_scaled())
    assert len(c) == 8
    counts = list(c.values())
    assert max(counts) - min(counts) <= 1  # even balance


def test_multiple_templates_for_grouped_cv():
    groups = {x.group for x in generate_verbfinal_scaled()}
    assert len(groups) >= 4


def test_roundtrip():
    x = generate_verbfinal_scaled()[0]
    assert VFStimulus(**x.to_dict()) == x
