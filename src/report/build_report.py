"""Build the opening scouting report: one self-contained HTML page.

    python -m src.report.build_report [--me <steamid>] [--output reports/opening_report.html]

"Me" is the player whose matches these are (default: the steamid present in the
most demos). Everyone on the other team in a demo counts as an enemy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineer import ZoneMapper, load_parsed

BUY_LABELS = {0: "eco", 1: "force", 2: "full"}
TEMPLATE = Path(__file__).with_name("template.html")


def buy_label(is_pistol, buy_type_code) -> str:
    return "pistol" if is_pistol else BUY_LABELS[int(buy_type_code)]


def detect_me(players: pd.DataFrame) -> str:
    per_demo = players.groupby("steamid")["demo_id"].nunique()
    return per_demo.idxmax()


def clean(v):
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return round(float(v), 1)
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def build_payload(processed: Path, parsed: Path, me: str | None) -> dict:
    players = pd.read_parquet(processed / "players.parquet")
    teams = pd.read_parquet(processed / "teams.parquet")
    # steamid64 exceeds JS number precision, so it travels as a string
    players["steamid"] = players["steamid"].astype(str)
    me = me or detect_me(players)
    me_name = players.loc[players["steamid"] == me, "name"].iloc[0] if (players["steamid"] == me).any() else me

    my_team = players[players["steamid"] == me].groupby("demo_id")["team_id"].first()
    players["is_enemy"] = (players["team_id"] != players["demo_id"].map(my_team)).astype(int)
    teams["is_enemy"] = (teams["team_id"] != teams["demo_id"].map(my_team)).astype(int)
    players["buy"] = [buy_label(p, b) for p, b in zip(players["is_pistol"], players["buy_type_code"])]
    teams["buy"] = [buy_label(p, b) for p, b in zip(teams["is_pistol"], teams["buy_type_code"])]
    teams["team_label"] = [n or p for n, p in zip(teams["team_name"], teams["players"])]

    # Trajectory dots (every 2 s) for the movement map, coloured by the player's route
    dots, centres = [], {}
    route_of = players.set_index(["demo_id", "round_number", "steamid"])[["route", "side", "is_enemy", "buy"]]
    by_map: dict[str, list] = {}
    for meta_path in sorted(parsed.glob("*/meta.json")):
        meta, _, _, traj = load_parsed(meta_path.parent)
        traj = traj[(traj["t_seconds"] % 2 == 0) & traj["is_alive"].astype(bool)].copy()
        traj["demo_id"] = meta["demo_id"]
        traj["steamid"] = traj["steamid"].astype(str)
        by_map.setdefault(meta["map_name"], []).append(traj)
    for map_name, frames in by_map.items():
        traj = pd.concat(frames)
        zones = ZoneMapper(map_name, traj)
        centres[map_name] = {k: [round(v[0]), round(v[1])] for k, v in zones.centres.items()}
        traj = traj.join(route_of, on=["demo_id", "round_number", "steamid"], how="inner")
        traj["map_name"] = map_name
        dots.append(traj[["map_name", "side", "route", "is_enemy", "buy", "t_seconds", "X", "Y"]])
    dots_df = pd.concat(dots)
    dots_df["X"] = dots_df["X"].round().astype(int)
    dots_df["Y"] = dots_df["Y"].round().astype(int)

    team_cols = ["demo_id", "map_name", "round_number", "side", "team_id", "team_label", "buy", "is_enemy",
                 "strategy", "n_A", "n_B", "n_Mid", "rushers", "rush_site", "plant_site", "won",
                 "first_kill_side", "first_kill_place", "first_kill_seconds"]
    player_cols = ["demo_id", "map_name", "round_number", "side", "steamid", "name", "team_id", "buy",
                   "is_enemy", "route", "rush", "site_arrival_seconds", "opening_line"]
    return {
        "me": {"steamid": me, "name": me_name},
        "summary": {
            "demos": int(players["demo_id"].nunique()),
            "rounds": int(teams[["demo_id", "round_number"]].drop_duplicates().shape[0]),
            "players": int(players["steamid"].nunique()),
        },
        "teams": records(teams[team_cols]),
        "players": records(players[player_cols]),
        "dots": dots_df.values.tolist(),
        "dot_columns": list(dots_df.columns),
        "centres": centres,
    }


def main():
    ap = argparse.ArgumentParser(description="Build the HTML opening report")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--parsed", default="data/parsed")
    ap.add_argument("--me", default=None, help="your steamid64 (default: auto-detect)")
    ap.add_argument("--output", default="reports/opening_report.html")
    args = ap.parse_args()

    payload = build_payload(Path(args.processed), Path(args.parsed), args.me)
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, default=clean))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Report: {out.resolve()}  (me = {payload['me']['name']})")


if __name__ == "__main__":
    main()
