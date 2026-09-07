"""Leitura de odds a partir de CSV (cola odds Betclic manualmente)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

# Colunas opcionais de mercados extra (além de 1X2 e O/U 2.5)
OPTIONAL_ODDS_KEYS = (
    "odds_over_25",
    "odds_under_25",
    "odds_over_15",
    "odds_under_15",
    "odds_over_35",
    "odds_under_35",
    "odds_btts_yes",
    "odds_btts_no",
    "odds_ah_m15",
    "odds_ah_m10",
    "odds_ah_m05",
    "odds_ah_p05",
    "odds_ah_p10",
    "odds_ah_p15",
    "odds_dc_1x",
    "odds_dc_12",
    "odds_dc_x2",
)


@dataclass(frozen=True)
class OddsRow:
    home: str
    away: str
    league: str
    odds_1: float
    odds_x: float
    odds_2: float
    odds_over_25: Optional[float] = None
    odds_under_25: Optional[float] = None
    odds_over_15: Optional[float] = None
    odds_under_15: Optional[float] = None
    odds_over_35: Optional[float] = None
    odds_under_35: Optional[float] = None
    odds_btts_yes: Optional[float] = None
    odds_btts_no: Optional[float] = None
    odds_ah_m15: Optional[float] = None
    odds_ah_m10: Optional[float] = None
    odds_ah_m05: Optional[float] = None
    odds_ah_p05: Optional[float] = None
    odds_ah_p10: Optional[float] = None
    odds_ah_p15: Optional[float] = None
    odds_dc_1x: Optional[float] = None
    odds_dc_12: Optional[float] = None
    odds_dc_x2: Optional[float] = None

    def odds_dict(self) -> dict[str, Optional[float]]:
        """Mapa chave CSV → odds (para iter_priced_markets)."""
        d: dict[str, Optional[float]] = {
            "odds_1": self.odds_1,
            "odds_x": self.odds_x,
            "odds_2": self.odds_2,
        }
        for key in OPTIONAL_ODDS_KEYS:
            d[key] = getattr(self, key)
        return d


REQUIRED = ("home", "away", "league", "odds_1", "odds_x", "odds_2")


def _f(val: str) -> float:
    return float(str(val).strip().replace(",", "."))


def _opt_f(val: Optional[str]) -> Optional[float]:
    if val is None or str(val).strip() == "":
        return None
    return _f(val)


def load_odds_csv(path: Path | str) -> list[OddsRow]:
    path = Path(path)
    rows: list[OddsRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV vazio ou sem cabeçalho")
        fields_map = {h.strip().lower(): h for h in reader.fieldnames}
        for req in REQUIRED:
            if req not in fields_map:
                raise ValueError(
                    f"Coluna obrigatória em falta: '{req}'. "
                    f"Cabeçalhos: {list(reader.fieldnames)}"
                )
        opt_field_names = {f.name for f in fields(OddsRow)} - {
            "home",
            "away",
            "league",
            "odds_1",
            "odds_x",
            "odds_2",
        }
        for i, raw in enumerate(reader, start=2):
            try:
                kwargs: dict = {
                    "home": raw[fields_map["home"]].strip(),
                    "away": raw[fields_map["away"]].strip(),
                    "league": raw[fields_map["league"]].strip(),
                    "odds_1": _f(raw[fields_map["odds_1"]]),
                    "odds_x": _f(raw[fields_map["odds_x"]]),
                    "odds_2": _f(raw[fields_map["odds_2"]]),
                }
                for name in opt_field_names:
                    if name in fields_map:
                        kwargs[name] = _opt_f(raw.get(fields_map[name]))
                    else:
                        kwargs[name] = None
                rows.append(OddsRow(**kwargs))
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"Linha {i} inválida: {exc}") from exc
    return rows
