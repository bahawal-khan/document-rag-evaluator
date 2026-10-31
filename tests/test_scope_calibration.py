import pytest

from core.generator import is_out_of_scope
from core.scope_calibration import calibrate_scope_threshold


def test_separable_scores_pick_the_midpoint():
    result = calibrate_scope_threshold([3, 4, 5, 2, 6, 3.5], [-9, -8, -10, -7, -9])
    assert result["separable"] is True
    assert -7 < result["threshold"] < 2
    assert result["in_scope_kept"] == 1.0
    assert result["off_topic_blocked"] == 1.0


def test_overlap_favours_answering_real_questions():
    result = calibrate_scope_threshold([-6, -5, -4, -3, 1, 2], [-5, -6, -4])
    assert result["separable"] is False
    assert result["in_scope_kept"] >= 0.8


def test_needs_enough_samples_and_ignores_bad_values():
    with pytest.raises(ValueError):
        calibrate_scope_threshold([1, 2], [])
    result = calibrate_scope_threshold([1, 2, 3, float("nan"), "x", None], [])
    assert result["in_scope_samples"] == 3


def test_is_out_of_scope_accepts_a_threshold_override():
    chunks = [{"rerank_score": -3.0}]
    assert is_out_of_scope(chunks, -2.0) is True
    assert is_out_of_scope(chunks, -5.0) is False
