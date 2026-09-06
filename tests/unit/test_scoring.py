import pytest

from fata.evaluation.scoring import score_mc, score_vqa, vqa_normalize


def test_vqa_normalization_and_consensus_without_substring_fallback():
    assert vqa_normalize("The TWO, cats!") == "2 cats"
    assert score_vqa("two", ["2", "two", "Two", "three"]) == 1.0
    assert score_vqa("cat on mat", ["cat", "cat", "cat"]) == 0.0


def test_vqa_empty_references_fail():
    with pytest.raises(ValueError):
        score_vqa("yes", [])


def test_mc_requires_option_letter_not_answer_substring():
    assert score_mc("Option C.", "c") == 1.0
    assert score_mc("I think the blue answer", "B") == 0.0
