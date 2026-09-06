import pytest

from fata.evaluation.metrics import compute_paper_metrics, retained_set_diagnostics


def test_metrics_keep_counts_and_do_not_clip_sr():
    # Fractional VQA scores affect accuracy; strict correctness uses score == 1.
    result = compute_paper_metrics(
        clean_full=[1, 0, .67, 1], attack_full=[1, 1, 1, 1],
        clean_practical=[1, 1, 0, 1], attack_practical=[0, 1, 0, 1],
    )
    assert result.sr_percent > 100
    assert (result.cbr_numerator, result.cbr_denominator) == (1, 2)
    assert (result.asr_numerator, result.asr_denominator) == (1, 3)
    assert (result.cc, result.cw, result.wc, result.ww) == (2, 0, 2, 0)
    assert result.net_harm == -2


def test_zero_full_accuracy_and_unaligned_vectors_fail():
    with pytest.raises(ZeroDivisionError):
        compute_paper_metrics([0], [0], [0], [0])
    with pytest.raises(ValueError):
        compute_paper_metrics([1, 0], [1], [1], [0])


def test_retained_set_jaccard_and_flip():
    result = retained_set_diagnostics([1, 2, 3], [2, 3, 4])
    assert result["jaccard"] == pytest.approx(.5)
    assert result["flip_rate"] == pytest.approx(.5)
