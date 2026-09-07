"""Forcas de ataque/defesa das equipas (dados de AMOSTRA offline)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_RATINGS = _DATA_DIR / "team_ratings.json"

LEAGUE_AVG_GOALS = {
    "primeira_liga": 2.55,
    "premier_league": 2.80,
    "la_liga": 2.60,
    "serie_a": 2.65,
    "bundesliga": 3.05,
}

LEAGUE_ALIASES = {
    "primeira": "primeira_liga",
    "primeira_liga": "primeira_liga",
    "portugal": "primeira_liga",
    "liga_portugal": "primeira_liga",
    "premier": "premier_league",
    "premier_league": "premier_league",
    "epl": "premier_league",
    "la_liga": "la_liga",
    "laliga": "la_liga",
    "spain": "la_liga",
    "serie_a": "serie_a",
    "seriea": "serie_a",
    "italy": "serie_a",
    "bundesliga": "bundesliga",
    "germany": "bundesliga",
}


@dataclass(frozen=True)
class TeamStrength:
    name: str
    attack: float
    defense: float
    league: str


class RatingsStore:
    """Carrega e consulta ratings de amostra por liga."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else _DEFAULT_RATINGS
        self._raw = self._load()
        self._by_league: dict[str, dict[str, TeamStrength]] = {}
        self._index: dict[str, TeamStrength] = {}
        self._build_index()

    def _load(self) -> dict:
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _build_index(self) -> None:
        for league, teams in self._raw.get("leagues", {}).items():
            league_map: dict[str, TeamStrength] = {}
            for name, vals in teams.items():
                ts = TeamStrength(
                    name=name,
                    attack=float(vals["attack"]),
                    defense=float(vals["defense"]),
                    league=league,
                )
                league_map[name.lower()] = ts
                self._index[name.lower()] = ts
            self._by_league[league] = league_map

    @staticmethod
    def normalize_league(league: Optional[str]) -> Optional[str]:
        if league is None:
            return None
        key = league.strip().lower().replace("-", "_").replace(" ", "_")
        return LEAGUE_ALIASES.get(key, key)

    def get(self, team: str, league: Optional[str] = None) -> TeamStrength:
        name = team.strip().lower()
        league_key = self.normalize_league(league)
        if league_key:
            league_map = self._by_league.get(league_key)
            if not league_map:
                raise KeyError(f"Liga desconhecida: {league}")
            if name not in league_map:
                raise KeyError(
                    f"Equipa '{team}' nao encontrada na liga '{league_key}'. "
                    f"Disponiveis: {', '.join(sorted(t.name for t in league_map.values()))}"
                )
            return league_map[name]
        if name not in self._index:
            raise KeyError(
                f"Equipa '{team}' nao encontrada nos ratings de amostra. "
                "Use --league ou confira data/team_ratings.json."
            )
        return self._index[name]

    def list_leagues(self) -> list[str]:
        return sorted(self._by_league.keys())

    def list_teams(self, league: Optional[str] = None) -> list[TeamStrength]:
        league_key = self.normalize_league(league)
        if league_key:
            return sorted(self._by_league[league_key].values(), key=lambda t: t.name)
        return sorted(self._index.values(), key=lambda t: (t.league, t.name))

    def league_avg_goals(self, league: str) -> float:
        key = self.normalize_league(league) or league
        return float(
            self._raw.get("league_avg_goals", {}).get(
                key, LEAGUE_AVG_GOALS.get(key, 2.7)
            )
        )

    def meta(self) -> dict:
        return {
            "label": self._raw.get("label", "sample"),
            "note": self._raw.get("note", ""),
            "leagues": self.list_leagues(),
            "metadata": self._raw.get("metadata", {}),
        }
