"""Value edge e Kelly fracionario (educacional)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValueBet:
    market: str
    model_prob: float
    odds: float
    implied_prob: float
    edge: float
    edge_pct: float
    kelly_full: float
    kelly_frac: float
    is_value: bool


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError(f"odds decimais invalidas: {decimal_odds}")
    return 1.0 / decimal_odds


def edge(model_prob: float, decimal_odds: float) -> float:
    """model_prob - implied_prob. Positivo = potencial value."""
    return model_prob - implied_probability(decimal_odds)


def kelly_fraction(
    model_prob: float, decimal_odds: float, fraction: float = 0.25
) -> tuple[float, float]:
    """
    Kelly classico para odds decimais: f* = (b*p - q) / b, b = odds-1, q=1-p.
    Retorna (kelly_full, kelly_frac). Negativos -> 0.
    """
    if decimal_odds <= 1.0 or not (0.0 <= model_prob <= 1.0):
        return 0.0, 0.0
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    full = (b * model_prob - q) / b
    full = max(0.0, full)
    frac = full * fraction
    return full, frac


def evaluate_value(
    market: str,
    model_prob: float,
    decimal_odds: float,
    min_edge: float = 0.04,
    kelly_scale: float = 0.25,
) -> ValueBet:
    """Avalia um mercado; min_edge default 4% (absoluto em probabilidade)."""
    impl = implied_probability(decimal_odds)
    ed = model_prob - impl
    full, frac = kelly_fraction(model_prob, decimal_odds, kelly_scale)
    return ValueBet(
        market=market,
        model_prob=model_prob,
        odds=decimal_odds,
        implied_prob=impl,
        edge=ed,
        edge_pct=ed * 100.0,
        kelly_full=full,
        kelly_frac=frac,
        is_value=ed >= min_edge and frac > 0,
    )


def rank_values(bets: list[ValueBet], only_value: bool = True) -> list[ValueBet]:
    filtered = [b for b in bets if b.is_value] if only_value else list(bets)
    return sorted(filtered, key=lambda b: b.edge, reverse=True)
