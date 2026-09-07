"""Mercados derivados da matriz de scores Poisson/Dixon-Coles.

Todos os mercados usam a mesma matriz P(home=i, away=j).

Asian Handicap (casa) — tratamento de push em linhas inteiras
--------------------------------------------------------------
Para linhas meias (-0.5, +0.5, -1.5, +1.5) não há empate de handicap:
  P(push)=0 e a probabilidade de cobertura é usada directamente no Kelly.

Para linhas inteiras (-1.0, +1.0, …): se o resultado com handicap for
exactamente 0, a casa de apostas **devolve a stake** (push).
Para value/edge/Kelly tratamos o push como *no-bet*:
  p_modelo = P(cover) / (1 - P(push))
(excluímos os outcomes de push; a stake não está em risco nesses casos).
Assim reutilizamos o Kelly binário win/lose existente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# Linhas AH casa suportadas (documentadas no README / CSV)
AH_HOME_LINES: tuple[float, ...] = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)


@dataclass(frozen=True)
class AsianHandicapResult:
    line: float
    p_win: float
    p_push: float
    p_lose: float
    """Probabilidade efectiva para value (exclui push)."""
    p_cover_for_value: float


def over_under_probs(mat: np.ndarray, line: float) -> tuple[float, float]:
    """
    Over/Under em linha .5 (1.5, 2.5, 3.5).
    Over = total golos > floor(line); Under = caso contrário.
    Ex.: 2.5 → Over se i+j >= 3.
    """
    threshold = int(line)  # 1.5 -> 1, 2.5 -> 2, 3.5 -> 3
    n = mat.shape[0]
    p_over = p_under = 0.0
    for i in range(n):
        for j in range(n):
            if i + j > threshold:
                p_over += mat[i, j]
            else:
                p_under += mat[i, j]
    return p_over, p_under


def btts_probs(mat: np.ndarray) -> tuple[float, float]:
    """Both teams to score: Yes / No."""
    n = mat.shape[0]
    p_yes = p_no = 0.0
    for i in range(n):
        for j in range(n):
            if i >= 1 and j >= 1:
                p_yes += mat[i, j]
            else:
                p_no += mat[i, j]
    return p_yes, p_no


def double_chance_probs(
    p_home: float, p_draw: float, p_away: float
) -> tuple[float, float, float]:
    """1X, 12, X2."""
    return p_home + p_draw, p_home + p_away, p_draw + p_away


def asian_handicap_home(mat: np.ndarray, line: float) -> AsianHandicapResult:
    """
    Asian handicap no lado da casa com linha `line` (ex.: -0.5, -1.0, +0.5).

    Resultado com handicap: (i - j + line).
      win  se > 0
      push se == 0  (só linhas inteiras)
      lose se < 0
    """
    n = mat.shape[0]
    p_win = p_push = p_lose = 0.0
    for i in range(n):
        for j in range(n):
            margin = (i - j) + line
            p = mat[i, j]
            # tolerância numérica para linhas inteiras
            if abs(margin) < 1e-12:
                p_push += p
            elif margin > 0:
                p_win += p
            else:
                p_lose += p
    non_push = p_win + p_lose
    if non_push <= 0:
        p_eff = 0.0
    else:
        p_eff = p_win / non_push
    return AsianHandicapResult(
        line=line,
        p_win=p_win,
        p_push=p_push,
        p_lose=p_lose,
        p_cover_for_value=p_eff,
    )


def ah_label(line: float) -> str:
    """Rótulo de mercado, ex.: AH-0.5, AH+1.0."""
    if line < 0:
        return f"AH{line:g}"
    return f"AH+{line:g}"


def ah_csv_key(line: float) -> str:
    """Chave CSV: odds_ah_m05, odds_ah_p10, …"""
    sign = "m" if line < 0 else "p"
    tenths = int(round(abs(line) * 10))
    return f"odds_ah_{sign}{tenths:02d}"


@dataclass(frozen=True)
class MarketProbs:
    """Todas as probs de mercado derivadas da matriz."""

    p_home: float
    p_draw: float
    p_away: float
    p_over_15: float
    p_under_15: float
    p_over_25: float
    p_under_25: float
    p_over_35: float
    p_under_35: float
    p_btts_yes: float
    p_btts_no: float
    p_dc_1x: float
    p_dc_12: float
    p_dc_x2: float
    ah: dict[float, AsianHandicapResult]


def compute_market_probs(mat: np.ndarray) -> MarketProbs:
    n = mat.shape[0]
    p_home = p_draw = p_away = 0.0
    for i in range(n):
        for j in range(n):
            p = mat[i, j]
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

    o15, u15 = over_under_probs(mat, 1.5)
    o25, u25 = over_under_probs(mat, 2.5)
    o35, u35 = over_under_probs(mat, 3.5)
    by, bn = btts_probs(mat)
    dc1x, dc12, dcx2 = double_chance_probs(p_home, p_draw, p_away)
    ah = {line: asian_handicap_home(mat, line) for line in AH_HOME_LINES}
    return MarketProbs(
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        p_over_15=o15,
        p_under_15=u15,
        p_over_25=o25,
        p_under_25=u25,
        p_over_35=o35,
        p_under_35=u35,
        p_btts_yes=by,
        p_btts_no=bn,
        p_dc_1x=dc1x,
        p_dc_12=dc12,
        p_dc_x2=dcx2,
        ah=ah,
    )


def iter_priced_markets(
    mp: MarketProbs,
    odds: dict[str, float | None],
) -> Iterable[tuple[str, float, float]]:
    """
    Gera (rótulo, p_modelo, odds) para cada mercado com odds presentes.

    Chaves esperadas em `odds` (opcionais excepto as usadas):
      odds_1, odds_x, odds_2,
      odds_over_15, odds_under_15, odds_over_25, odds_under_25,
      odds_over_35, odds_under_35,
      odds_btts_yes, odds_btts_no,
      odds_ah_m15, odds_ah_m10, odds_ah_m05, odds_ah_p05, odds_ah_p10, odds_ah_p15,
      odds_dc_1x, odds_dc_12, odds_dc_x2
    """

    def _add(label: str, p: float, key: str) -> tuple[str, float, float] | None:
        o = odds.get(key)
        if o is None:
            return None
        return label, p, float(o)

    pairs: list[tuple[str, float, str]] = [
        ("1", mp.p_home, "odds_1"),
        ("X", mp.p_draw, "odds_x"),
        ("2", mp.p_away, "odds_2"),
        ("O1.5", mp.p_over_15, "odds_over_15"),
        ("U1.5", mp.p_under_15, "odds_under_15"),
        ("O2.5", mp.p_over_25, "odds_over_25"),
        ("U2.5", mp.p_under_25, "odds_under_25"),
        ("O3.5", mp.p_over_35, "odds_over_35"),
        ("U3.5", mp.p_under_35, "odds_under_35"),
        ("BTTS_Y", mp.p_btts_yes, "odds_btts_yes"),
        ("BTTS_N", mp.p_btts_no, "odds_btts_no"),
        ("DC_1X", mp.p_dc_1x, "odds_dc_1x"),
        ("DC_12", mp.p_dc_12, "odds_dc_12"),
        ("DC_X2", mp.p_dc_x2, "odds_dc_x2"),
    ]
    for label, p, key in pairs:
        item = _add(label, p, key)
        if item is not None:
            yield item

    for line in AH_HOME_LINES:
        key = ah_csv_key(line)
        item = _add(ah_label(line), mp.ah[line].p_cover_for_value, key)
        if item is not None:
            yield item
