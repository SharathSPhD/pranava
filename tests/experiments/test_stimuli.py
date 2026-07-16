"""TDD: the controlled stimulus set is well-formed and balanced."""
from collections import Counter

from pranava.experiments.stimuli import generate_stimuli, Stimulus


def test_min_size():
    s = generate_stimuli()
    assert len(s) >= 200


def test_unique_ids():
    s = generate_stimuli()
    assert len({x.id for x in s}) == len(s)


def test_disambig_index_within_bounds():
    for x in generate_stimuli():
        assert 0 <= x.disambig_word_index < x.n_words, x


def test_resolution_classes_present_and_balanced():
    s = generate_stimuli()
    c = Counter(x.resolution for x in s)
    assert c["early"] > 0 and c["late"] > 0
    # neither class should be less than 20% of the set
    frac_late = c["late"] / len(s)
    assert 0.2 <= frac_late <= 0.8


def test_late_items_disambiguate_late():
    # late-resolving items must disambiguate in the back half of the sentence
    for x in generate_stimuli():
        if x.resolution == "late":
            assert x.disambig_word_index >= x.n_words / 2, x


def test_early_items_disambiguate_early():
    for x in generate_stimuli():
        if x.resolution == "early":
            assert x.disambig_word_index <= x.n_words / 2, x


def test_structures_present():
    s = generate_stimuli()
    structs = set(x.structure for x in s)
    assert {"canonical", "garden_path", "verb_final", "verb_first"} <= structs


def test_deterministic():
    a = [x.id + x.text for x in generate_stimuli()]
    b = [x.id + x.text for x in generate_stimuli()]
    assert a == b


def test_roundtrip_dict():
    x = generate_stimuli()[0]
    assert Stimulus(**x.to_dict()) == x
