"""Testes de calibração com CSV sintético mínimo."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from betting_algo.calibrate import (
    TEAM_NAME_MAP,
    calibrate_from_csv_rows,
    calibrate_league,
    map_team_name,
    parse_matches,
    run_calibration,
)
from betting_algo.poisson import expected_goals
from betting_algo.ratings import TeamStrength


def _synthetic_rows() -> list[dict]:
    """
    Liga de 3 equipas A (forte), B (média), C (fraca).
    A marca mais e concede menos; C o contrário.
    """
    # Round-robin home/away com scores enviesados
    return [
        {"Date": "01/01/2025", "HomeTeam": "Alpha", "AwayTeam": "Beta", "FTHG": 3, "FTAG": 1},
        {"Date": "02/01/2025", "HomeTeam": "Beta", "AwayTeam": "Alpha", "FTHG": 0, "FTAG": 2},
        {"Date": "03/01/2025", "HomeTeam": "Alpha", "AwayTeam": "Gamma", "FTHG": 4, "FTAG": 0},
        {"Date": "04/01/2025", "HomeTeam": "Gamma", "AwayTeam": "Alpha", "FTHG": 0, "FTAG": 3},
        {"Date": "05/01/2025", "HomeTeam": "Beta", "AwayTeam": "Gamma", "FTHG": 2, "FTAG": 1},
        {"Date": "06/01/2025", "HomeTeam": "Gamma", "AwayTeam": "Beta", "FTHG": 1, "FTAG": 1},
        # segunda volta
        {"Date": "07/01/2025", "HomeTeam": "Alpha", "AwayTeam": "Beta", "FTHG": 2, "FTAG": 0},
        {"Date": "08/01/2025", "HomeTeam": "Beta", "AwayTeam": "Alpha", "FTHG": 1, "FTAG": 2},
        {"Date": "09/01/2025", "HomeTeam": "Alpha", "AwayTeam": "Gamma", "FTHG": 3, "FTAG": 1},
        {"Date": "10/01/2025", "HomeTeam": "Gamma", "AwayTeam": "Alpha", "FTHG": 0, "FTAG": 2},
        {"Date": "11/01/2025", "HomeTeam": "Beta", "AwayTeam": "Gamma", "FTHG": 2, "FTAG": 0},
        {"Date": "12/01/2025", "HomeTeam": "Gamma", "AwayTeam": "Beta", "FTHG": 0, "FTAG": 2},
    ]


def test_map_team_name_known_aliases():
    assert map_team_name("Sp Lisbon") == "Sporting"
    assert map_team_name("Man City") == "Manchester City"
    assert map_team_name("Benfica") == "Benfica"


def test_calibrate_strong_team_has_higher_attack():
    cal = calibrate_from_csv_rows(_synthetic_rows(), league="toy")
    assert set(cal.teams) == {"Alpha", "Beta", "Gamma"}
    assert cal.teams["Alpha"]["attack"] > cal.teams["Beta"]["attack"]
    assert cal.teams["Beta"]["attack"] > cal.teams["Gamma"]["attack"]
    # Alpha concede menos → defense rating mais baixo
    assert cal.teams["Alpha"]["defense"] < cal.teams["Gamma"]["defense"]
    assert cal.league_avg_goals > 0
    assert cal.n_matches == 12


def test_calibrate_attack_defense_mean_near_one():
    cal = calibrate_from_csv_rows(_synthetic_rows(), league="toy")
    atts = [v["attack"] for v in cal.teams.values()]
    defs = [v["defense"] for v in cal.teams.values()]
    assert sum(atts) / len(atts) == pytest.approx(1.0, abs=0.02)
    assert sum(defs) / len(defs) == pytest.approx(1.0, abs=0.02)


def test_calibrated_lambdas_track_observed_means():
    """Forças calibradas devem ordenar favoritos de forma coerente com o Poisson."""
    rows = _synthetic_rows()
    cal = calibrate_from_csv_rows(rows, league="toy", home_advantage=1.10)
    store = {
        n: TeamStrength(name=n, attack=v["attack"], defense=v["defense"], league="toy")
        for n, v in cal.teams.items()
    }
    # Alpha em casa vs Gamma deve ter λ_casa bem acima de λ_fora
    lam_h, lam_a = expected_goals(
        store["Alpha"], store["Gamma"], cal.league_avg_goals, home_advantage=1.10
    )
    assert lam_h > lam_a
    assert lam_h > 1.5
    # Gamma em casa vs Alpha: ainda assim Alpha (fora) competitivo / Gamma fraco
    lam_h2, lam_a2 = expected_goals(
        store["Gamma"], store["Alpha"], cal.league_avg_goals, home_advantage=1.10
    )
    assert lam_a2 > lam_h2


def test_parse_matches_and_run_offline(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # escrever mini CSVs no formato football-data para uma liga
    # Usamos P1 code via naming; run_calibration espera códigos reais.
    # Em vez disso testamos parse + calibrate_league e escrita manual.
    fixture = tmp_path / "toy.csv"
    with open(fixture, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        w.writeheader()
        for r in _synthetic_rows():
            w.writerow(r)
    matches = parse_matches(fixture, "toy", "2425")
    assert len(matches) == 12
    cal = calibrate_league(matches, "toy")
    assert "Alpha" in cal.teams

    # Pipeline offline com ficheiros nomeados como época_código
    for code, league in [("P1", "primeira_liga"), ("E0", "premier_league"),
                         ("SP1", "la_liga"), ("I1", "serie_a"), ("D1", "bundesliga")]:
        path = raw / f"2425_{code}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
            )
            w.writeheader()
            # renomear Alpha→ nomes distintos por liga para evitar colisões no índice global
            for r in _synthetic_rows():
                row = dict(r)
                row["HomeTeam"] = f"{league[:3]}_{r['HomeTeam']}"
                row["AwayTeam"] = f"{league[:3]}_{r['AwayTeam']}"
                w.writerow(row)

    out = tmp_path / "team_ratings.json"
    payload = run_calibration(
        seasons=["2425"],
        download=False,
        raw_dir=raw,
        out_path=out,
    )
    assert out.exists()
    assert payload["label"] == "calibrated"
    assert "metadata" in payload
    assert payload["metadata"]["seasons"] == ["2425"]
    assert "calibrated_at" in payload["metadata"]
    assert "source" in payload["metadata"]
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["leagues"]) == 5


def test_team_name_map_nonempty():
    assert "Sp Lisbon" in TEAM_NAME_MAP
    assert TEAM_NAME_MAP["Sp Braga"] == "Braga"
