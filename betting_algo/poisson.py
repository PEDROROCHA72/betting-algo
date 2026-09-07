"""Modelo de score: Poisson independente com ajuste leve estilo Dixon-Coles."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from typing import Optional

import numpy as np

from betting_algo.markets import MarketProbs, compute_market_probs
from betting_algo.ratings import RatingsStore, TeamStrength


@dataclass(frozen=True)
class MatchPrediction:
    home: str
    away: str
    league: str
    lambda_home: float
    lambda_away: float
    p_home: float
    p_draw: float
    p_away: float
    p_over_25: float
    p_under_25: float
    # Mercados extra (mesma matriz)
    p_over_15: float
    p_under_15: float
    p_over_35: float
    p_under_35: float
    p_btts_yes: float
    p_btts_no: float
    p_dc_1x: float
    p_dc_12: float
    p_dc_x2: float
    markets: MarketProbs
    score_matrix: np.ndarray


def _poisson_pmf(k: int, lam: float) -> float:
    if lam < 0:
        raise ValueError("lambda deve ser >= 0")
    return exp(-lam) * (lam**k) / factorial(k)


def expected_goals(
    home: TeamStrength,
    away: TeamStrength,
    league_avg: float,
    home_advantage: float = 1.10,
) -> tuple[float, float]:
    """
    Ratings ~1.0 = media da liga; defense baixo = defesa forte (concede menos).
    lambda_home = attack_home * defense_away * (league_avg/2) * home_adv
    """
    lambda_home = home.attack * away.defense * (league_avg / 2.0) * home_advantage
    lambda_away = away.attack * home.defense * (league_avg / 2.0)
    return max(0.05, lambda_home), max(0.05, lambda_away)


def dixon_coles_tau(
    i: int, j: int, lam_h: float, lam_a: float, rho: float
) -> float:
    """Correcao Dixon-Coles para scores baixos (0-0, 1-0, 0-1, 1-1)."""
    if i == 0 and j == 0:
        return 1.0 - lam_h * lam_a * rho
    if i == 0 and j == 1:
        return 1.0 + lam_h * rho
    if i == 1 and j == 0:
        return 1.0 + lam_a * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(
    lam_h: float,
    lam_a: float,
    max_goals: int = 8,
    rho: float = -0.05,
) -> np.ndarray:
    """Matriz P(home=i, away=j) com Poisson independente + tau Dixon-Coles leve."""
    mat = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for i in range(max_goals + 1):
        ph = _poisson_pmf(i, lam_h)
        for j in range(max_goals + 1):
            pa = _poisson_pmf(j, lam_a)
            tau = dixon_coles_tau(i, j, lam_h, lam_a, rho)
            mat[i, j] = max(0.0, ph * pa * tau)
    total = mat.sum()
    if total <= 0:
        raise RuntimeError("matriz de scores invalida")
    return mat / total


def probs_from_matrix(mat: np.ndarray) -> tuple[float, float, float, float, float]:
    """Retorna (p_home, p_draw, p_away, p_over_25, p_under_25). Compatibilidade."""
    mp = compute_market_probs(mat)
    return mp.p_home, mp.p_draw, mp.p_away, mp.p_over_25, mp.p_under_25


def predict_match(
    home_name: str,
    away_name: str,
    store: Optional[RatingsStore] = None,
    league: Optional[str] = None,
    home_advantage: float = 1.10,
    max_goals: int = 8,
    rho: float = -0.05,
) -> MatchPrediction:
    store = store or RatingsStore()
    home = store.get(home_name, league)
    away = store.get(away_name, league or home.league)
    if home.league != away.league and league is None:
        league_avg = (
            store.league_avg_goals(home.league) + store.league_avg_goals(away.league)
        ) / 2
        league_label = f"{home.league}/{away.league}"
    else:
        league_key = store.normalize_league(league) or home.league
        league_avg = store.league_avg_goals(league_key)
        league_label = league_key

    lam_h, lam_a = expected_goals(home, away, league_avg, home_advantage)
    mat = score_matrix(lam_h, lam_a, max_goals=max_goals, rho=rho)
    mp = compute_market_probs(mat)
    return MatchPrediction(
        home=home.name,
        away=away.name,
        league=league_label,
        lambda_home=lam_h,
        lambda_away=lam_a,
        p_home=mp.p_home,
        p_draw=mp.p_draw,
        p_away=mp.p_away,
        p_over_25=mp.p_over_25,
        p_under_25=mp.p_under_25,
        p_over_15=mp.p_over_15,
        p_under_15=mp.p_under_15,
        p_over_35=mp.p_over_35,
        p_under_35=mp.p_under_35,
        p_btts_yes=mp.p_btts_yes,
        p_btts_no=mp.p_btts_no,
        p_dc_1x=mp.p_dc_1x,
        p_dc_12=mp.p_dc_12,
        p_dc_x2=mp.p_dc_x2,
        markets=mp,
        score_matrix=mat,
    )
