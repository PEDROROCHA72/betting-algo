"""Smoke tests de integração com dados sample."""

from pathlib import Path

import pytest

from betting_algo.cli import main
from betting_algo.markets import iter_priced_markets
from betting_algo.odds_io import load_odds_csv
from betting_algo.poisson import predict_match
from betting_algo.ratings import RatingsStore
from betting_algo.value import evaluate_value

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_odds.csv"


def test_ratings_load():
    store = RatingsStore()
    assert "primeira_liga" in store.list_leagues()
    t = store.get("Benfica", "primeira_liga")
    assert t.attack > 1.0


def test_predict_benfica_porto():
    pred = predict_match("Benfica", "Porto", league="primeira_liga")
    assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 1e-8
    assert pred.lambda_home > 0 and pred.lambda_away > 0
    assert pred.p_btts_yes + pred.p_btts_no == pytest.approx(1.0, abs=1e-8)
    assert pred.p_over_15 >= pred.p_over_25 >= pred.p_over_35


def test_sample_csv_loads_new_columns():
    rows = load_odds_csv(SAMPLE)
    assert len(rows) >= 5
    assert any(r.odds_btts_yes is not None for r in rows)
    assert any(r.odds_ah_m05 is not None for r in rows)
    assert any(r.odds_over_15 is not None for r in rows)


def test_sample_csv_loads_and_scans():
    rows = load_odds_csv(SAMPLE)
    assert len(rows) >= 5
    store = RatingsStore()
    hits = 0
    new_hits = 0
    for row in rows:
        pred = predict_match(row.home, row.away, store=store, league=row.league)
        for market, p, o in iter_priced_markets(pred.markets, row.odds_dict()):
            vb = evaluate_value(market, p, o, min_edge=0.04)
            if vb.is_value:
                hits += 1
                if market.startswith(("BTTS", "AH", "O1.5", "O3.5", "U1.5", "U3.5", "DC_")):
                    new_hits += 1
    assert hits >= 1
    assert new_hits >= 1


def test_demo_cli_runs():
    assert main(["demo", "--edge", "0.04"]) == 0


def test_scan_cli_runs():
    assert main(["scan", str(SAMPLE), "--edge", "0.04"]) == 0
