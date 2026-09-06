import math
import pytest

from fata.evaluation.kprac import select_llava_clean_threshold_kprac


def trajectory(**updates):
    value = {576: 1.0, 192: .99, 128: .95, 64: .81, 32: .79, 16: .2}
    value.update({int(k): v for k, v in updates.items()})
    return value


def test_selects_smallest_numeric_clean_only_candidate():
    result = select_llava_clean_threshold_kprac(
        dataset="d", compressor="c", clean_accuracy_by_k=trajectory(),
        sample_count=1000, candidates=[192, 16, 64, 32, 128],
    )
    assert result.selected_k == 64
    assert result.retention == pytest.approx(.81)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_nan_or_inf_fails(bad):
    with pytest.raises(ValueError):
        select_llava_clean_threshold_kprac(
            dataset="d", compressor="c", clean_accuracy_by_k=trajectory(**{"32": bad}),
            sample_count=1,
        )


def test_missing_duplicate_zero_denominator_and_no_match_fail():
    cases = []
    missing = trajectory(); del missing[16]
    cases.append((missing, [192, 128, 64, 32, 16], ValueError))
    cases.append((trajectory(), [192, 128, 64, 32, 32], ValueError))
    zero = trajectory(); zero[576] = 0
    cases.append((zero, [192, 128, 64, 32, 16], ZeroDivisionError))
    low = {576: 1, 192: .7, 128: .6, 64: .5, 32: .4, 16: .3}
    cases.append((low, [192, 128, 64, 32, 16], ValueError))
    for values, candidates, error in cases:
        with pytest.raises(error):
            select_llava_clean_threshold_kprac(
                dataset="d", compressor="c", clean_accuracy_by_k=values,
                sample_count=10, candidates=candidates,
            )
