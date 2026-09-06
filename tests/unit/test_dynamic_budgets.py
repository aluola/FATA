import pytest

from fata.compression.budgets import actual_keep_count, internvl_budget_weights
from fata.constants import practical_fraction


def test_dynamic_rounding_and_protocol_distinction():
    assert actual_keep_count(256, 1, 9) == 28
    assert actual_keep_count(256, 1, 18) == 14
    assert actual_keep_count(1, 1, 36) == 1
    assert practical_fraction("TextVQA_Open") == pytest.approx(1 / 9)
    assert practical_fraction("VQAv2_MC") == pytest.approx(1 / 18)


def test_internvl_task_specific_weights():
    assert internvl_budget_weights("TextVQA_Open")["2/9"] == .25
    assert internvl_budget_weights("ScienceQA_MC")["1/18"] == 1.0


@pytest.mark.parametrize("args", [(0, 1, 9), (10, 0, 9), (10, 10, 9)])
def test_invalid_budget_inputs(args):
    with pytest.raises(ValueError):
        actual_keep_count(*args)
