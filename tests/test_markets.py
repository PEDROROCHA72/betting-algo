"""Testes dos mercados derivados da matriz (BTTS, O/U, AH, DC)."""

import pytest

from betting_algo.markets import (
    asian_handicap_home,
    btts_probs,
    compute_market_probs,
    double_chance_probs,
    over_under_probs,
)
from betting_algo.poisson import score_matrix


@pytest.fixture
def mat_balanced():
    return score_matrix(1.4, 1.2, max_goals=8, rho=-0.05)


@pytest.fixture
def mat_home_fav():
    return score_matrix(2.0, 0.9, max_goals=8, rho=-0.05)


def test_btts_probs_sum_to_one(mat_balanced):
    yes, no = btts_probs(mat_balanced)
    assert yes + no == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < yes < 1.0


def test_ou_lines_monotonic(mat_balanced):
    o15, _ = over_under_probs(mat_balanced, 1.5)
    o25, _ = over_under_probs(mat_balanced, 2.5)
    o35, _ = over_under_probs(mat_balanced, 3.5)
    assert o15 >= o25 >= o35
    assert o15 + over_under_probs(mat_balanced, 1.5)[1] == pytest.approx(1.0, abs=1e-9)


def test_ah_half_line_equals_home_win(mat_home_fav):
    mp = compute_market_probs(mat_home_fav)
    ah = asian_handicap_home(mat_home_fav, -0.5)
    assert ah.p_push == pytest.approx(0.0, abs=1e-12)
    assert ah.p_win == pytest.approx(mp.p_home, abs=1e-9)
    assert ah.p_cover_for_value == pytest.approx(mp.p_home, abs=1e-9)


def test_ah_plus_half_equals_home_or_draw(mat_home_fav):
    mp = compute_market_probs(mat_home_fav)
    ah = asian_handicap_home(mat_home_fav, 0.5)
    assert ah.p_push == pytest.approx(0.0, abs=1e-12)
    assert ah.p_win == pytest.approx(mp.p_home + mp.p_draw, abs=1e-9)


def test_ah_integer_push_accounted(mat_home_fav):
    """AH -1.0: push quando casa vence por exactamente 1."""
    ah = asian_handicap_home(mat_home_fav, -1.0)
    assert ah.p_push > 0.05  # probabilidade material de vencer por 1
    assert ah.p_win + ah.p_push + ah.p_lose == pytest.approx(1.0, abs=1e-9)
    # p_cover_for_value exclui push
    assert ah.p_cover_for_value == pytest.approx(
        ah.p_win / (ah.p_win + ah.p_lose), abs=1e-9
    )
    assert ah.p_cover_for_value > ah.p_win  # excluir push aumenta a prob efectiva


def test_ah_plus_one_push(mat_balanced):
    ah = asian_handicap_home(mat_balanced, 1.0)
    assert ah.p_push > 0.0
    assert ah.p_win + ah.p_push + ah.p_lose == pytest.approx(1.0, abs=1e-9)


def test_double_chance(mat_balanced):
    mp = compute_market_probs(mat_balanced)
    d1x, d12, dx2 = double_chance_probs(mp.p_home, mp.p_draw, mp.p_away)
    assert d1x == pytest.approx(mp.p_home + mp.p_draw)
    assert d12 == pytest.approx(mp.p_home + mp.p_away)
    assert dx2 == pytest.approx(mp.p_draw + mp.p_away)
    assert d1x + dx2 - mp.p_draw == pytest.approx(1.0, abs=1e-9)


def test_compute_market_probs_consistency(mat_balanced):
    mp = compute_market_probs(mat_balanced)
    assert mp.p_home + mp.p_draw + mp.p_away == pytest.approx(1.0, abs=1e-9)
    assert mp.p_btts_yes + mp.p_btts_no == pytest.approx(1.0, abs=1e-9)
    assert mp.p_over_15 >= mp.p_over_25 >= mp.p_over_35
