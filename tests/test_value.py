"""Testes de edge e Kelly."""

import pytest

from betting_algo.value import (
    edge,
    evaluate_value,
    implied_probability,
    kelly_fraction,
    rank_values,
)


def test_implied_probability():
    assert implied_probability(2.0) == pytest.approx(0.5)
    assert implied_probability(4.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        implied_probability(1.0)


def test_edge_positive_when_model_above_implied():
    # modelo 40%, odds 3.0 => implied 33.3% => edge ~6.7%
    ed = edge(0.40, 3.0)
    assert ed == pytest.approx(0.40 - 1 / 3.0)
    assert ed > 0.04


def test_edge_negative_when_overpriced():
    ed = edge(0.30, 2.0)  # implied 50%
    assert ed < 0


def test_kelly_zero_when_no_edge():
    full, frac = kelly_fraction(0.40, 2.0, fraction=0.25)  # fair-ish below
    # p=0.4, odds=2 => b=1, f=(1*0.4-0.6)/1 = -0.2 -> 0
    assert full == 0.0
    assert frac == 0.0


def test_kelly_positive_and_fractional():
    # p=0.5, odds=2.5 => b=1.5, f=(1.5*0.5 - 0.5)/1.5 = 0.25/1.5 ≈ 0.1667
    full, frac = kelly_fraction(0.50, 2.5, fraction=0.25)
    assert full == pytest.approx(0.25 / 1.5)
    assert frac == pytest.approx(full * 0.25)


def test_evaluate_value_flags_min_edge():
    # edge = 0.40 - 1/3 ≈ 0.0667 >= 0.04
    vb = evaluate_value("1", 0.40, 3.0, min_edge=0.04, kelly_scale=0.25)
    assert vb.is_value is True
    assert vb.edge_pct == pytest.approx(vb.edge * 100)
    assert vb.kelly_frac > 0

    vb2 = evaluate_value("X", 0.34, 3.0, min_edge=0.04, kelly_scale=0.25)
    # edge ≈ 0.0067 < 0.04
    assert vb2.is_value is False


def test_rank_values_sorts_by_edge():
    a = evaluate_value("1", 0.45, 3.0, min_edge=0.04)
    b = evaluate_value("2", 0.50, 2.5, min_edge=0.04)
    ranked = rank_values([a, b], only_value=True)
    assert ranked[0].edge >= ranked[-1].edge
