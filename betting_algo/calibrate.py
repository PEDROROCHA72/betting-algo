"""Calibração de attack/defense a partir de resultados históricos (football-data.co.uk).

Modelo alinhado com poisson.expected_goals:
  λ_casa = attack_casa × defense_fora × (league_avg/2) × home_advantage
  λ_fora = attack_fora × defense_casa × (league_avg/2)

Ratings ~1.0 = média da liga; defense < 1 = defesa mais forte.
"""

from __future__ import annotations

import csv
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_RAW_DIR = _DATA_DIR / "raw"
_DEFAULT_OUT = _DATA_DIR / "team_ratings.json"

# football-data.co.uk division codes → chave interna
LEAGUE_CODES: dict[str, str] = {
    "P1": "primeira_liga",
    "E0": "premier_league",
    "SP1": "la_liga",
    "I1": "serie_a",
    "D1": "bundesliga",
}

LEAGUE_TO_CODE = {v: k for k, v in LEAGUE_CODES.items()}

# Preferir host sem www (www por vezes devolve 503)
_BASE_URLS = (
    "https://football-data.co.uk/mmz4281/{season}/{code}.csv",
    "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv",
)

DEFAULT_SEASONS = ("2526", "2425")
MIN_MATCHES_FOR_SEASON = 8  # época incompleta / vazia se abaixo
HOME_ADVANTAGE = 1.10  # igual ao default em poisson.predict_match
ITERATIONS = 40
EPS = 1e-9

# Nomes CSV (football-data) → nomes usados em sample_odds / ratings de amostra
TEAM_NAME_MAP: dict[str, str] = {
    # Premier League
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    # Primeira Liga
    "Sp Lisbon": "Sporting",
    "Sp Braga": "Braga",
    "Guimaraes": "Vitoria Guimaraes",
    # La Liga
    "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Sociedad": "Real Sociedad",
    # Bundesliga
    "Leverkusen": "Bayer Leverkusen",
    "Ein Frankfurt": "Frankfurt",
    "M'gladbach": "Monchengladbach",
    "FC Koln": "Koln",
}


@dataclass
class MatchResult:
    date: str
    home: str
    away: str
    fthg: int
    ftag: int
    season: str
    league: str


@dataclass
class LeagueCalibration:
    league: str
    league_avg_goals: float
    avg_home_goals: float
    avg_away_goals: float
    home_advantage: float
    teams: dict[str, dict[str, float]]  # name -> {attack, defense}
    n_matches: int
    seasons: list[str]
    renames_applied: dict[str, str] = field(default_factory=dict)


def map_team_name(raw: str) -> str:
    name = raw.strip()
    return TEAM_NAME_MAP.get(name, name)


def season_url(season: str, code: str, base: Optional[str] = None) -> str:
    template = base or _BASE_URLS[0]
    return template.format(season=season, code=code)


def _looks_like_csv(content: bytes) -> bool:
    head = content[:200].lstrip().lower()
    if head.startswith(b"<html") or b"temporarily unavailable" in head:
        return False
    text = content[:500].decode("utf-8", errors="replace")
    return "HomeTeam" in text or "FTHG" in text


def download_csv(
    season: str,
    code: str,
    dest_dir: Path,
    timeout: float = 30.0,
) -> Path:
    """Descarrega CSV para dest_dir/{season}_{code}.csv. Levanta RuntimeError se falhar."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{season}_{code}.csv"
    last_err: Optional[Exception] = None
    for base in _BASE_URLS:
        url = season_url(season, code, base)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; betting-algo/0.1)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not _looks_like_csv(data):
                last_err = RuntimeError(f"resposta inválida de {url}")
                continue
            out.write_bytes(data)
            return out
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_err = exc
            continue
    raise RuntimeError(
        f"Falha ao descarregar {season}/{code}: {last_err}"
    )


def ensure_downloads(
    seasons: Iterable[str],
    codes: Optional[Iterable[str]] = None,
    raw_dir: Path = _RAW_DIR,
    force: bool = False,
) -> list[Path]:
    """Garante CSVs em raw_dir; descarrega se em falta ou force=True."""
    codes = list(codes or LEAGUE_CODES.keys())
    paths: list[Path] = []
    for season in seasons:
        for code in codes:
            path = raw_dir / f"{season}_{code}.csv"
            if force or not path.exists() or path.stat().st_size < 100:
                path = download_csv(season, code, raw_dir)
            paths.append(path)
    return paths


def parse_matches(
    path: Path,
    league: str,
    season: str,
) -> list[MatchResult]:
    """Lê Date, HomeTeam, AwayTeam, FTHG, FTAG; ignora linhas sem golos."""
    matches: list[MatchResult] = []
    renames: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return matches
        # normalizar cabeçalhos
        field_map = {h.strip(): h for h in reader.fieldnames if h}
        required = ("HomeTeam", "AwayTeam", "FTHG", "FTAG")
        for req in required:
            if req not in field_map:
                raise ValueError(f"{path}: coluna em falta '{req}'")
        date_key = field_map.get("Date")
        for row in reader:
            raw_h = (row.get(field_map["HomeTeam"]) or "").strip()
            raw_a = (row.get(field_map["AwayTeam"]) or "").strip()
            gh = (row.get(field_map["FTHG"]) or "").strip()
            ga = (row.get(field_map["FTAG"]) or "").strip()
            if not raw_h or not raw_a or gh == "" or ga == "":
                continue
            try:
                fthg = int(float(gh))
                ftag = int(float(ga))
            except ValueError:
                continue
            home = map_team_name(raw_h)
            away = map_team_name(raw_a)
            if home != raw_h:
                renames[raw_h] = home
            if away != raw_a:
                renames[raw_a] = away
            date = (row.get(date_key) or "").strip() if date_key else ""
            matches.append(
                MatchResult(
                    date=date,
                    home=home,
                    away=away,
                    fthg=fthg,
                    ftag=ftag,
                    season=season,
                    league=league,
                )
            )
    # stash renames on path attribute via return; caller aggregates
    parse_matches.last_renames = renames  # type: ignore[attr-defined]
    return matches


parse_matches.last_renames = {}  # type: ignore[attr-defined]


def calibrate_league(
    matches: list[MatchResult],
    league: str,
    home_advantage: float = HOME_ADVANTAGE,
    iterations: int = ITERATIONS,
) -> LeagueCalibration:
    """Estima attack/defense por equipa via iteração relativa (estilo Maher)."""
    if not matches:
        raise ValueError(f"Sem jogos para calibrar liga {league}")

    seasons = sorted({m.season for m in matches})
    avg_home = sum(m.fthg for m in matches) / len(matches)
    avg_away = sum(m.ftag for m in matches) / len(matches)
    league_avg = avg_home + avg_away
    # Mesmas bases que poisson.expected_goals (att=def=1)
    mu_h = (league_avg / 2.0) * home_advantage
    mu_a = league_avg / 2.0

    teams = sorted({m.home for m in matches} | {m.away for m in matches})
    attack = {t: 1.0 for t in teams}
    defense = {t: 1.0 for t in teams}

    # Pré-indexar jogos por equipa
    home_games: dict[str, list[MatchResult]] = {t: [] for t in teams}
    away_games: dict[str, list[MatchResult]] = {t: [] for t in teams}
    for m in matches:
        home_games[m.home].append(m)
        away_games[m.away].append(m)

    for _ in range(iterations):
        new_att: dict[str, float] = {}
        for t in teams:
            scored = 0.0
            expected_den = 0.0
            for m in home_games[t]:
                scored += m.fthg
                expected_den += defense[m.away] * mu_h
            for m in away_games[t]:
                scored += m.ftag
                expected_den += defense[m.home] * mu_a
            new_att[t] = scored / max(expected_den, EPS)
        # normalizar média aritmética = 1 (alinha λ médios com golos observados)
        am = sum(new_att.values()) / len(new_att)
        attack = {t: new_att[t] / max(am, EPS) for t in teams}

        new_def: dict[str, float] = {}
        for t in teams:
            conceded = 0.0
            expected_den = 0.0
            for m in home_games[t]:
                conceded += m.ftag
                expected_den += attack[m.away] * mu_a
            for m in away_games[t]:
                conceded += m.fthg
                expected_den += attack[m.home] * mu_h
            new_def[t] = conceded / max(expected_den, EPS)
        am_d = sum(new_def.values()) / len(new_def)
        defense = {t: new_def[t] / max(am_d, EPS) for t in teams}

    team_out = {
        t: {
            "attack": round(attack[t], 4),
            "defense": round(defense[t], 4),
        }
        for t in teams
    }
    return LeagueCalibration(
        league=league,
        league_avg_goals=round(league_avg, 4),
        avg_home_goals=round(avg_home, 4),
        avg_away_goals=round(avg_away, 4),
        home_advantage=home_advantage,
        teams=team_out,
        n_matches=len(matches),
        seasons=seasons,
    )


def discover_usable_seasons(
    preferred: Iterable[str],
    raw_dir: Path = _RAW_DIR,
    codes: Optional[Iterable[str]] = None,
    min_matches: int = MIN_MATCHES_FOR_SEASON,
) -> list[str]:
    """Devolve épocas com dados suficientes (pelo menos num código)."""
    codes = list(codes or LEAGUE_CODES.keys())
    usable: list[str] = []
    for season in preferred:
        ok = False
        for code in codes:
            path = raw_dir / f"{season}_{code}.csv"
            if not path.exists():
                continue
            try:
                league = LEAGUE_CODES[code]
                matches = parse_matches(path, league, season)
                if len(matches) >= min_matches:
                    ok = True
                    break
            except (ValueError, OSError):
                continue
        if ok:
            usable.append(season)
    return usable


def load_league_matches(
    league: str,
    seasons: Iterable[str],
    raw_dir: Path = _RAW_DIR,
) -> tuple[list[MatchResult], dict[str, str]]:
    code = LEAGUE_TO_CODE[league]
    all_matches: list[MatchResult] = []
    renames: dict[str, str] = {}
    for season in seasons:
        path = raw_dir / f"{season}_{code}.csv"
        if not path.exists():
            continue
        matches = parse_matches(path, league, season)
        renames.update(getattr(parse_matches, "last_renames", {}))
        all_matches.extend(matches)
    return all_matches, renames


def build_ratings_payload(
    calibrations: dict[str, LeagueCalibration],
    seasons_used: list[str],
) -> dict:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    league_avg = {lg: cal.league_avg_goals for lg, cal in calibrations.items()}
    leagues = {lg: cal.teams for lg, cal in calibrations.items()}
    all_renames: dict[str, str] = {}
    match_counts = {}
    for lg, cal in calibrations.items():
        all_renames.update(cal.renames_applied)
        match_counts[lg] = {"n_matches": cal.n_matches, "n_teams": len(cal.teams)}

    return {
        "label": "calibrated",
        "note": (
            "Ratings calibrados a partir de resultados reais "
            f"(épocas {', '.join(seasons_used)}). "
            "Fonte: football-data.co.uk. "
            "Attack/defense ~1.0 = média da liga; defense < 1 = defesa mais forte."
        ),
        "metadata": {
            "source": "football-data.co.uk",
            "source_url_pattern": "https://football-data.co.uk/mmz4281/{season}/{code}.csv",
            "seasons": seasons_used,
            "calibrated_at": now,
            "home_advantage": HOME_ADVANTAGE,
            "method": "iterative_relative_strength",
            "iterations": ITERATIONS,
            "league_codes": LEAGUE_CODES,
            "team_renames": all_renames,
            "match_counts": match_counts,
        },
        "league_avg_goals": league_avg,
        "leagues": leagues,
    }


def run_calibration(
    seasons: Optional[list[str]] = None,
    download: bool = False,
    raw_dir: Path = _RAW_DIR,
    out_path: Path = _DEFAULT_OUT,
    home_advantage: float = HOME_ADVANTAGE,
) -> dict:
    """Pipeline completo: (download) → parse → calibrate → escrever JSON."""
    preferred = list(seasons) if seasons else list(DEFAULT_SEASONS)
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)

    if download:
        # tenta cada época; falhas individuais não abortam se já houver ficheiros
        for season in preferred:
            for code in LEAGUE_CODES:
                path = raw_dir / f"{season}_{code}.csv"
                try:
                    download_csv(season, code, raw_dir)
                except RuntimeError as exc:
                    if not path.exists():
                        print(f"[aviso] {exc}")

    usable = discover_usable_seasons(preferred, raw_dir=raw_dir)
    if not usable:
        # fallback: qualquer época presente no disco
        found = sorted(
            {
                p.name.split("_")[0]
                for p in raw_dir.glob("*_*.csv")
                if p.name[:4].isdigit()
            },
            reverse=True,
        )
        usable = discover_usable_seasons(found, raw_dir=raw_dir)
    if not usable:
        raise FileNotFoundError(
            f"Nenhum CSV utilizável em {raw_dir}. "
            "Corra com --download ou coloque ficheiros {{season}}_{{code}}.csv."
        )

    calibrations: dict[str, LeagueCalibration] = {}
    for league in LEAGUE_CODES.values():
        matches, renames = load_league_matches(league, usable, raw_dir=raw_dir)
        if len(matches) < 10:
            print(f"[aviso] liga {league}: poucos jogos ({len(matches)}), a saltar")
            continue
        cal = calibrate_league(matches, league, home_advantage=home_advantage)
        cal.renames_applied = renames
        calibrations[league] = cal

    if not calibrations:
        raise RuntimeError("Calibração não produziu nenhuma liga")

    payload = build_ratings_payload(calibrations, usable)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def print_summary(payload: dict) -> None:
    meta = payload.get("metadata", {})
    print(f"label={payload.get('label')} | seasons={meta.get('seasons')} | "
          f"calibrated_at={meta.get('calibrated_at')}")
    print(f"fonte: {meta.get('source')}")
    avgs = payload.get("league_avg_goals", {})
    leagues = payload.get("leagues", {})
    for league, teams in leagues.items():
        print(f"\n=== {league} ===")
        print(f"  equipas={len(teams)}  avg_goals={avgs.get(league)}  "
              f"matches={meta.get('match_counts', {}).get(league, {}).get('n_matches')}")
        ranked = sorted(teams.items(), key=lambda kv: kv[1]["attack"], reverse=True)
        top = ranked[:3]
        bottom = ranked[-3:]
        print("  top attack:", ", ".join(f"{n}={v['attack']:.3f}" for n, v in top))
        print("  bottom attack:", ", ".join(f"{n}={v['attack']:.3f}" for n, v in bottom))
    renames = meta.get("team_renames") or {}
    if renames:
        print("\nRenames CSV → interno:")
        for raw, mapped in sorted(renames.items()):
            print(f"  {raw!r} → {mapped!r}")


def calibrate_from_csv_rows(
    rows: list[dict],
    league: str = "test_league",
    home_advantage: float = HOME_ADVANTAGE,
) -> LeagueCalibration:
    """Helper para testes: rows com HomeTeam, AwayTeam, FTHG, FTAG."""
    matches = [
        MatchResult(
            date=r.get("Date", ""),
            home=map_team_name(str(r["HomeTeam"])),
            away=map_team_name(str(r["AwayTeam"])),
            fthg=int(r["FTHG"]),
            ftag=int(r["FTAG"]),
            season=r.get("Season", "test"),
            league=league,
        )
        for r in rows
    ]
    return calibrate_league(matches, league, home_advantage=home_advantage)
