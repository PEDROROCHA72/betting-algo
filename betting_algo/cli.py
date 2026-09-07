"""CLI: demo | predict | scan | rate | calibrate"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from betting_algo import __version__
from betting_algo.markets import AH_HOME_LINES, ah_label, iter_priced_markets
from betting_algo.odds_io import load_odds_csv
from betting_algo.poisson import predict_match
from betting_algo.ratings import RatingsStore
from betting_algo.value import evaluate_value, rank_values

_DATA = Path(__file__).resolve().parent.parent / "data"
_SAMPLE_ODDS = _DATA / "sample_odds.csv"
_RAW = _DATA / "raw"
_RATINGS = _DATA / "team_ratings.json"

DEFAULT_EDGE = 0.04
DEFAULT_KELLY = 0.25


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_pred_header(pred) -> str:
    mp = pred.markets
    lines = [
        f"{pred.home} vs {pred.away}  [{pred.league}]",
        f"  λ casa={pred.lambda_home:.2f}  λ fora={pred.lambda_away:.2f}",
        f"  1X2:     1={_pct(pred.p_home)}  X={_pct(pred.p_draw)}  2={_pct(pred.p_away)}",
        f"  DC:      1X={_pct(pred.p_dc_1x)}  12={_pct(pred.p_dc_12)}  X2={_pct(pred.p_dc_x2)}",
        f"  BTTS:    Yes={_pct(pred.p_btts_yes)}  No={_pct(pred.p_btts_no)}",
        (
            f"  O/U:     O1.5={_pct(pred.p_over_15)} U1.5={_pct(pred.p_under_15)} | "
            f"O2.5={_pct(pred.p_over_25)} U2.5={_pct(pred.p_under_25)} | "
            f"O3.5={_pct(pred.p_over_35)} U3.5={_pct(pred.p_under_35)}"
        ),
    ]
    ah_parts = []
    for line in AH_HOME_LINES:
        ah = mp.ah[line]
        if ah.p_push > 1e-12:
            ah_parts.append(
                f"{ah_label(line)}={_pct(ah.p_cover_for_value)}"
                f"(push={_pct(ah.p_push)})"
            )
        else:
            ah_parts.append(f"{ah_label(line)}={_pct(ah.p_cover_for_value)}")
    lines.append("  AH casa: " + "  ".join(ah_parts))
    return "\n".join(lines)


def _markets_from_odds_dict(pred, odds: dict) -> list[tuple[str, float, float]]:
    return list(iter_priced_markets(pred.markets, odds))


def _print_values(bets, title: str, only_value: bool = True) -> None:
    ranked = rank_values(bets, only_value=only_value)
    print(f"\n=== {title} ===")
    if not ranked:
        print("  (nenhuma oportunidade com edge >= limiar)")
        return
    print(
        f"{'Mercado':<10} {'P_modelo':>9} {'Odds':>7} {'Impl.':>8} "
        f"{'Edge%':>7} {'Kelly*':>8} {'Value?':>7}"
    )
    for b in ranked:
        flag = "SIM" if b.is_value else "-"
        print(
            f"{b.market:<10} {_pct(b.model_prob):>9} {b.odds:>7.2f} {_pct(b.implied_prob):>8} "
            f"{b.edge_pct:>6.1f}% {b.kelly_frac:>7.2%} {flag:>7}"
        )
    print("  * Kelly fracionário (educacional; default 1/4 Kelly) — sem garantia.")
    print(
        "  * AH linhas inteiras: P_modelo exclui push (stake devolvida = no-bet)."
    )


def _fmt_odds_snippet(odds: dict) -> str:
    parts = [f"1={odds['odds_1']:.2f}", f"X={odds['odds_x']:.2f}", f"2={odds['odds_2']:.2f}"]
    extras = [
        ("O1.5", "odds_over_15"),
        ("U1.5", "odds_under_15"),
        ("O2.5", "odds_over_25"),
        ("U2.5", "odds_under_25"),
        ("O3.5", "odds_over_35"),
        ("U3.5", "odds_under_35"),
        ("BTTS_Y", "odds_btts_yes"),
        ("BTTS_N", "odds_btts_no"),
        ("AH-0.5", "odds_ah_m05"),
        ("AH-1", "odds_ah_m10"),
        ("AH+0.5", "odds_ah_p05"),
        ("AH+1", "odds_ah_p10"),
        ("AH-1.5", "odds_ah_m15"),
        ("AH+1.5", "odds_ah_p15"),
        ("1X", "odds_dc_1x"),
        ("12", "odds_dc_12"),
        ("X2", "odds_dc_x2"),
    ]
    for label, key in extras:
        v = odds.get(key)
        if v is not None:
            parts.append(f"{label}={v:.2f}")
    return "  ".join(parts)


def cmd_demo(args: argparse.Namespace) -> int:
    store = RatingsStore()
    meta = store.meta()
    print(f"betting_algo v{__version__} — DEMO (dados: {meta.get('label', 'sample')})")
    print(meta.get("note", ""))
    print(f"Limiar de edge: {args.edge*100:.0f}% | Kelly frac: {args.kelly}")
    print("AVISO: análise educativa. Apostas envolvem risco de perda. Sem garantia.\n")

    rows = load_odds_csv(args.csv or _SAMPLE_ODDS)
    all_hits = []
    for row in rows:
        try:
            pred = predict_match(row.home, row.away, store=store, league=row.league)
        except KeyError as exc:
            print(f"[skip] {row.home} vs {row.away}: {exc}")
            continue
        odds = row.odds_dict()
        print(_fmt_pred_header(pred))
        print(f"  Odds CSV: {_fmt_odds_snippet(odds)}")
        markets = _markets_from_odds_dict(pred, odds)
        bets = [
            evaluate_value(m, p, o, min_edge=args.edge, kelly_scale=args.kelly)
            for m, p, o in markets
        ]
        for b in bets:
            if b.is_value:
                all_hits.append((f"{pred.home} vs {pred.away}", b))
        _print_values(bets, "Mercados deste jogo", only_value=False)
        print()

    print("=" * 60)
    print("OPORTUNIDADES RANQUEADAS (edge >= limiar)")
    print("=" * 60)
    if not all_hits:
        print("  Nenhuma value bet no sample com o limiar atual.")
        return 0
    print(
        f"{'Jogo':<36} {'Mkt':<10} {'P_mod':>7} {'Odds':>6} {'Edge%':>7} {'Kelly*':>8}"
    )
    ranked = sorted(all_hits, key=lambda t: t[1].edge, reverse=True)
    for label, b in ranked:
        print(
            f"{label:<36} {b.market:<10} {_pct(b.model_prob):>7} {b.odds:>6.2f} "
            f"{b.edge_pct:>6.1f}% {b.kelly_frac:>7.2%}"
        )
    print("\n* Stake Kelly fracionário = sugestão educativa (não conselho financeiro).")
    return 0


def _predict_odds_from_args(args: argparse.Namespace) -> dict[str, Optional[float]]:
    return {
        "odds_1": args.odds_1,
        "odds_x": args.odds_x,
        "odds_2": args.odds_2,
        "odds_over_15": getattr(args, "odds_over_15", None),
        "odds_under_15": getattr(args, "odds_under_15", None),
        "odds_over_25": args.odds_over,
        "odds_under_25": args.odds_under,
        "odds_over_35": getattr(args, "odds_over_35", None),
        "odds_under_35": getattr(args, "odds_under_35", None),
        "odds_btts_yes": getattr(args, "odds_btts_yes", None),
        "odds_btts_no": getattr(args, "odds_btts_no", None),
        "odds_ah_m15": getattr(args, "odds_ah_m15", None),
        "odds_ah_m10": getattr(args, "odds_ah_m10", None),
        "odds_ah_m05": getattr(args, "odds_ah_m05", None),
        "odds_ah_p05": getattr(args, "odds_ah_p05", None),
        "odds_ah_p10": getattr(args, "odds_ah_p10", None),
        "odds_ah_p15": getattr(args, "odds_ah_p15", None),
        "odds_dc_1x": getattr(args, "odds_dc_1x", None),
        "odds_dc_12": getattr(args, "odds_dc_12", None),
        "odds_dc_x2": getattr(args, "odds_dc_x2", None),
    }


def cmd_predict(args: argparse.Namespace) -> int:
    store = RatingsStore()
    pred = predict_match(args.home, args.away, store=store, league=args.league)
    print(_fmt_pred_header(pred))
    odds = _predict_odds_from_args(args)
    has_1x2 = args.odds_1 and args.odds_x and args.odds_2
    has_any = any(v is not None for v in odds.values())
    if has_1x2 or has_any:
        # Se faltar 1X2 completo mas houver outros mercados, avaliar o que existir
        markets = _markets_from_odds_dict(pred, odds)
        if not markets:
            print("\n(Nenhuma odd válida para avaliar.)")
            return 0
        bets = [
            evaluate_value(m, p, o, min_edge=args.edge, kelly_scale=args.kelly)
            for m, p, o in markets
        ]
        _print_values(bets, "Value vs odds fornecidas", only_value=False)
    else:
        print(
            "\n(Dica: passe --odds-1/--odds-x/--odds-2 e/ou flags de outros mercados "
            "para calcular edge e Kelly.)"
        )
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    store = RatingsStore()
    rows = load_odds_csv(args.path)
    all_hits = []
    for row in rows:
        try:
            pred = predict_match(row.home, row.away, store=store, league=row.league)
        except KeyError as exc:
            print(f"[skip] {row.home} vs {row.away}: {exc}", file=sys.stderr)
            continue
        markets = _markets_from_odds_dict(pred, row.odds_dict())
        for m, p, o in markets:
            b = evaluate_value(m, p, o, min_edge=args.edge, kelly_scale=args.kelly)
            if b.is_value:
                all_hits.append((f"{pred.home} vs {pred.away}", b))
    print(f"Scan: {args.path} | edge>={args.edge*100:.0f}% | kelly={args.kelly}")
    if not all_hits:
        print("Nenhuma value bet encontrada.")
        return 0
    print(
        f"{'Jogo':<36} {'Mkt':<10} {'P_mod':>7} {'Odds':>6} {'Edge%':>7} {'Kelly*':>8}"
    )
    for label, b in sorted(all_hits, key=lambda t: t[1].edge, reverse=True):
        print(
            f"{label:<36} {b.market:<10} {_pct(b.model_prob):>7} {b.odds:>6.2f} "
            f"{b.edge_pct:>6.1f}% {b.kelly_frac:>7.2%}"
        )
    return 0


def cmd_rate(args: argparse.Namespace) -> int:
    store = RatingsStore()
    meta = store.meta()
    print(f"Ratings: {meta.get('label')} — {meta.get('note')}")
    teams = store.list_teams(args.league)
    if args.league:
        print(f"Liga: {store.normalize_league(args.league)}")
    print(f"{'Equipa':<28} {'Liga':<16} {'Attack':>8} {'Defense':>8}")
    for t in teams:
        print(f"{t.name:<28} {t.league:<16} {t.attack:>8.2f} {t.defense:>8.2f}")
    print("\nNota: defense < 1.0 = defesa mais forte (concede menos).")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from betting_algo.calibrate import print_summary, run_calibration

    seasons = None
    if args.seasons:
        seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    try:
        payload = run_calibration(
            seasons=seasons,
            download=args.download,
            raw_dir=Path(args.raw_dir) if args.raw_dir else _RAW,
            out_path=Path(args.out) if args.out else _RATINGS,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    print(f"Escrito: {args.out or _RATINGS}")
    print_summary(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="betting_algo",
        description="Análise de value bets no futebol (MVP educativo, contexto Betclic PT).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Correr demo com sample_odds.csv")
    demo.add_argument("--csv", type=Path, default=None, help="CSV alternativo")
    demo.add_argument("--edge", type=float, default=DEFAULT_EDGE, help="Limiar de edge (default 0.04)")
    demo.add_argument("--kelly", type=float, default=DEFAULT_KELLY, help="Fração Kelly (default 0.25)")
    demo.set_defaults(func=cmd_demo)

    pred = sub.add_parser(
        "predict",
        help="Prever mercados (1X2, BTTS, O/U, AH, DC) para HOME AWAY",
    )
    pred.add_argument("home")
    pred.add_argument("away")
    pred.add_argument("--league", default=None)
    pred.add_argument("--odds-1", type=float, default=None)
    pred.add_argument("--odds-x", type=float, default=None)
    pred.add_argument("--odds-2", type=float, default=None)
    pred.add_argument("--odds-over", type=float, default=None, help="Over 2.5")
    pred.add_argument("--odds-under", type=float, default=None, help="Under 2.5")
    pred.add_argument("--odds-over-15", type=float, default=None)
    pred.add_argument("--odds-under-15", type=float, default=None)
    pred.add_argument("--odds-over-35", type=float, default=None)
    pred.add_argument("--odds-under-35", type=float, default=None)
    pred.add_argument("--odds-btts-yes", type=float, default=None)
    pred.add_argument("--odds-btts-no", type=float, default=None)
    pred.add_argument("--odds-ah-m15", type=float, default=None, help="AH casa -1.5")
    pred.add_argument("--odds-ah-m10", type=float, default=None, help="AH casa -1.0")
    pred.add_argument("--odds-ah-m05", type=float, default=None, help="AH casa -0.5")
    pred.add_argument("--odds-ah-p05", type=float, default=None, help="AH casa +0.5")
    pred.add_argument("--odds-ah-p10", type=float, default=None, help="AH casa +1.0")
    pred.add_argument("--odds-ah-p15", type=float, default=None, help="AH casa +1.5")
    pred.add_argument("--odds-dc-1x", type=float, default=None)
    pred.add_argument("--odds-dc-12", type=float, default=None)
    pred.add_argument("--odds-dc-x2", type=float, default=None)
    pred.add_argument("--edge", type=float, default=DEFAULT_EDGE)
    pred.add_argument("--kelly", type=float, default=DEFAULT_KELLY)
    pred.set_defaults(func=cmd_predict)

    scan = sub.add_parser("scan", help="Varrer CSV de odds e listar value bets")
    scan.add_argument("path", type=Path)
    scan.add_argument("--edge", type=float, default=DEFAULT_EDGE)
    scan.add_argument("--kelly", type=float, default=DEFAULT_KELLY)
    scan.set_defaults(func=cmd_scan)

    rate = sub.add_parser("rate", help="Listar ratings (amostra ou calibrados)")
    rate.add_argument("--league", default=None)
    rate.set_defaults(func=cmd_rate)

    cal = sub.add_parser(
        "calibrate",
        help="Calibrar attack/defense a partir de CSVs football-data.co.uk",
    )
    cal.add_argument(
        "--seasons",
        default=None,
        help="Épocas CSV separados por vírgula (ex.: 2526,2425). Default: 2526 depois 2425",
    )
    cal.add_argument(
        "--download",
        action="store_true",
        help="Descarregar CSVs para data/raw/ antes de calibrar",
    )
    cal.add_argument("--raw-dir", type=Path, default=None, help="Pasta dos CSVs brutos")
    cal.add_argument("--out", type=Path, default=None, help="Destino team_ratings.json")
    cal.set_defaults(func=cmd_calibrate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
