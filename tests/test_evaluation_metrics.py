"""Tests for :mod:`chimera_rag.evaluation.metrics`."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------
def test_exact_match_case_and_punctuation_insensitive():
    from chimera_rag.evaluation.metrics import exact_match

    assert exact_match("Alfred Nobel", "alfred nobel") == 1.0
    assert exact_match("Alfred Nobel.", "Alfred Nobel") == 1.0
    assert exact_match("The Eiffel Tower", "eiffel tower") == 1.0  # strip articles


def test_exact_match_returns_zero_for_unrelated():
    from chimera_rag.evaluation.metrics import exact_match

    assert exact_match("Einstein", "Newton") == 0.0


# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------
def test_token_f1_perfect_overlap_is_one():
    from chimera_rag.evaluation.metrics import token_f1

    assert token_f1("Alfred Nobel", "Alfred Nobel") == pytest.approx(1.0)


def test_token_f1_partial_overlap_is_between_0_and_1():
    from chimera_rag.evaluation.metrics import token_f1

    # ref = {alfred, nobel}, pred = {alfred}
    # precision=1, recall=0.5, f1 = 2*1*0.5/(1+0.5) = 0.666...
    assert token_f1(pred="Alfred", ref="Alfred Nobel") == pytest.approx(2 / 3)


def test_token_f1_no_overlap_is_zero():
    from chimera_rag.evaluation.metrics import token_f1

    assert token_f1("Einstein", "Nobel") == 0.0


def test_token_f1_empty_strings_return_zero():
    from chimera_rag.evaluation.metrics import token_f1

    assert token_f1("", "") == 0.0
    assert token_f1("a", "") == 0.0
