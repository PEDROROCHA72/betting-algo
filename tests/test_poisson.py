"""Testes das probabilidades Poisson / Dixon-Coles."""

import numpy as np
import pytest

from betting_algo.poisson import (
    dixon_coles_tau,
    probs_from_matrix,
    score_matrix,
    _poisson_pmf,
)


def test_poisson_pmf_sums_roughly_to_one():
    lam = 1.3
    s = sum(_poisson_pmf(k, lam) for k in range(0, 20))
    assert s == pytest.approx(1.0, abs=1e-6)


def test_score_matrix_normalized():
    mat = score_matrix(1.4, 1.1, max_goals=8, rho=-0.05)
    assert mat.shape == (9, 9)
    assert mat.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(mat >= 0)


def test_probs_1x2_sum_to_one():
    mat = score_matrix(1.5, 1.0, max_goals=8, rho=-0.05)
    p_h, p_d, p_a, p_o, p_u = probs_from_matrix(mat)
    assert p_h + p_d + p_a == pytest.approx(1.0, abs=1e-9)
    assert p_o + p_u == pytest.approx(1.0, abs=1e-9)
    # Casa favorita com lambda maior
    assert p_h > p_a


def test_strong_home_favorite():
    mat = score_matrix(2.2, 0.7, max_goals=8, rho=-0.05)
    p_h, p_d, p_a, _, _ = probs_from_matrix(mat)
    assert p_h > 0.55
    assert p_a < 0.20


def test_dixon_coles_tau_low_scores():
    assert dixon_coles_tau(1, 1, 1.2, 1.0, -0.05) == pytest.approx(1.05)
    assert dixon_coles_tau(2, 2, 1.2, 1.0, -0.05) == 1.0


def test_over_25_increases_with_lambdas():
    low = score_matrix(0.8, 0.7, max_goals=8, rho=0.0)
    high = score_matrix(2.0, 1.8, max_goals=8, rho=0.0)
    _, _, _, o_low, _ = probs_from_matrix(low)
    _, _, _, o_high, _ = probs_from_matrix(high)
    assert o_high > o_low
