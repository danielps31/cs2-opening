"""Build opening datasets from parsed demos.

players dataset - one row per player per round:
  features (schema.FEATURE_COLUMNS), team_id, route (A/B/Mid), rush, site_arrival_seconds,
  opening_line (first places visited), x_<t>/y_<t> position targets
teams dataset - one row per team per round:
  T side: players to A/B/Mid, rushers, plant site, strategy label (e.g. "B rush")
  CT side: setup like "2A-1Mid-2B"

A player's zone comes from, in order:
  1. a bombsite place name (BombsiteA / BombsiteB)
  2. the map's zone list in configs/maps.yaml (exact place name)
  3. geometry: clearly closer to one site's centre -> that site, otherwise Mid
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import map_zones, model_config
from src.data.demo_parser import place_site
from src.features.schema import add_derived_features

logger = logging.getLogger(__name__)

LOSS_BONUS_BASE = 1400
LOSS_BONUS_STEP = 500
LOSS_BONUS_MAX_LEVEL = 4  # 1400 .. 3400
SPAWN_PLACES = {"TSpawn", "CTSpawn", ""}
SITE_RATIO = 0.6  # distance to nearer site / farther site below this -> counts as that site


class ZoneMapper:
    """Place name + position -> A / B / Mid for one map."""

    def __init__(self, map_name: str, samples: pd.DataFrame):
        self.exact = {
            place.lower(): route
            for route, places in map_zones(map_name).items()
            for place in places
        }
        self.centres = {}
        for site in ("A", "B"):
            at_site = samples[samples["last_place_name"].map(place_site) == site]
            if not at_site.empty:
                self.centres[site] = (at_site["X"].mean(), at_site["Y"].mean())

    def zone(self, place, x: float, y: float) -> str | None:
        site = place_site(place)
        if site:
            return site
        if not isinstance(place, str) or place in SPAWN_PLACES:
            return None
        if place.lower() in self.exact:
            return self.exact[place.lower()]
        if len(self.centres) < 2 or np.isnan(x):
            return None
        da = np.hypot(x - self.centres["A"][0], y - self.centres["A"][1])
        db = np.hypot(x - self.centres["B"][0], y - self.centres["B"][1])
        if min(da, db) / max(da, db) < SITE_RATIO:
            return "A" if da < db else "B"
        return "Mid"


def team_assignment(players: pd.DataFrame) -> pd.DataFrame:
    """Which team (0/1) each player belongs to. Team 0 = T side in the first round."""
    first_round = players["round_number"].min()
    first = players[players["round_number"] == first_round]
    team = {sid: (0 if side == "T" else 1) for sid, side in zip(first["steamid"], first["side"])}
    # Players who joined later: same side as team 0 that round?
    out = players[["round_number", "steamid", "side"]].copy()
    out["team"] = out["steamid"].map(team)
    team0_side = out[out["team"] == 0].groupby("round_number")["side"].agg(lambda s: s.mode().iloc[0])
    missing = out["team"].isna()
    out.loc[missing, "team"] = np.where(
        out.loc[missing, "side"] == out.loc[missing, "round_number"].map(team0_side), 0, 1)
    return out[["round_number", "steamid", "team"]].astype({"team": int})


def round_context(rounds: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Score, previous winner and loss bonus per (round, team), before the round is played."""
    teams = players.merge(team_assignment(players), on=["round_number", "steamid"])
    team0_side = teams[teams["team"] == 0].groupby("round_number")["side"].agg(lambda s: s.mode().iloc[0])
    rounds = rounds.assign(team0_side=rounds["round_number"].map(team0_side)).dropna(subset=["team0_side"])

    score = [0, 0]
    loss_level = [0, 0]
    prev_winner_team = None
    prev_team0_side = None
    rows = []
    for r in rounds.sort_values("round_number").itertuples():
        if prev_team0_side is not None and r.team0_side != prev_team0_side:
            loss_level = [0, 0]  # new half: economy reset
        side_of = {r.team0_side: 0, ("CT" if r.team0_side == "T" else "T"): 1}
        for side, team in side_of.items():
            rows.append({
                "round_number": r.round_number,
                "side": side,
                "team": team,
                "score_team": score[team],
                "score_opponent": score[1 - team],
                "prev_round_won": int(prev_winner_team == team),
                "loss_bonus": LOSS_BONUS_BASE + LOSS_BONUS_STEP * loss_level[team],
            })
        if r.winner in side_of:
            w = side_of[r.winner]
            score[w] += 1
            # CS2: a loss raises the loss level, a win lowers it by one
            loss_level[1 - w] = min(loss_level[1 - w] + 1, LOSS_BONUS_MAX_LEVEL)
            loss_level[w] = max(loss_level[w] - 1, 0)
            prev_winner_team = w
        else:
            prev_winner_team = None
        prev_team0_side = r.team0_side
    return pd.DataFrame(rows)


def opening_line(places: list, max_steps: int = 6) -> str:
    """Distinct consecutive places, e.g. 'TSpawn > Middle > TopofMid > BombsiteA'.

    Long paths keep the start and the final place.
    """
    line = []
    for p in places:
        if isinstance(p, str) and p and (not line or line[-1] != p):
            line.append(p)
    if len(line) > max_steps:
        line = line[:max_steps - 1] + [line[-1]]
    return " > ".join(line)


def player_openings(players: pd.DataFrame, trajectory: pd.DataFrame, zones: ZoneMapper) -> pd.DataFrame:
    """Route, rush and opening line per player per round."""
    labels = model_config()["labels"]
    rush_s, ct_setup_s, horizon = labels["rush_seconds"], labels["ct_setup_seconds"], labels["route_horizon_seconds"]

    traj = trajectory[(trajectory["t_seconds"] <= horizon) & trajectory["is_alive"].astype(bool)]
    traj = traj.merge(players[["round_number", "steamid", "side"]], on=["round_number", "steamid"])
    traj = traj.sort_values("t_seconds")
    traj["zone"] = [zones.zone(p, x, y) for p, x, y in zip(traj["last_place_name"], traj["X"], traj["Y"])]
    traj["site"] = traj["last_place_name"].map(place_site)

    rows = []
    for (rnd, sid), g in traj.groupby(["round_number", "steamid"], sort=False):
        side = g["side"].iloc[0]
        on_site = g[g["site"].notna()]
        row = {"round_number": rnd, "steamid": sid, "route": "unknown", "rush": 0, "site_arrival_seconds": np.nan}
        if side == "T":
            if not on_site.empty:
                first = on_site.iloc[0]
                row["route"] = first["site"]
                row["site_arrival_seconds"] = first["t_seconds"]
                row["rush"] = int(first["t_seconds"] <= rush_s)
                path = g[g["t_seconds"] <= first["t_seconds"]]
            else:
                committed = g["zone"].dropna()
                row["route"] = committed.iloc[-1] if not committed.empty else "unknown"
                path = g
        else:
            setup = g[g["t_seconds"] <= ct_setup_s]["zone"].dropna()
            row["route"] = setup.iloc[-1] if not setup.empty else "unknown"
            path = g[g["t_seconds"] <= ct_setup_s]
        row["opening_line"] = opening_line(path["last_place_name"].tolist())
        rows.append(row)
    return pd.DataFrame(rows)


def team_strategy(group: pd.DataFrame, plant_site) -> dict:
    counts = group["route"].value_counts()
    n_a, n_b, n_mid = (int(counts.get(k, 0)) for k in ("A", "B", "Mid"))
    out = {"n_A": n_a, "n_B": n_b, "n_Mid": n_mid}
    if group["side"].iloc[0] == "CT":
        out["strategy"] = f"{n_a}A-{n_mid}Mid-{n_b}B"
        out["rushers"] = 0
        out["rush_site"] = None
        return out

    rushers = group[group["rush"] == 1]
    rush_site = rushers["route"].mode().iloc[0] if not rushers.empty else None
    out["rushers"] = len(rushers)
    out["rush_site"] = rush_site
    if len(rushers) >= 3:
        out["strategy"] = f"{rush_site} rush"
    elif max(n_a, n_b) >= 3:
        out["strategy"] = f"{'A' if n_a >= n_b else 'B'} execute"
    elif isinstance(plant_site, str):
        out["strategy"] = f"Default > {plant_site}"
    else:
        out["strategy"] = "Default"
    return out


def team_id(steamids) -> str:
    return hashlib.sha1(",".join(sorted(map(str, steamids))).encode()).hexdigest()[:10]


def position_targets(trajectory: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    traj = trajectory[trajectory["t_seconds"].isin(horizons)].copy()
    traj.loc[~traj["is_alive"].astype(bool), ["X", "Y"]] = np.nan
    wide = traj.pivot_table(index=["round_number", "steamid"], columns="t_seconds", values=["X", "Y"], dropna=False)
    wide.columns = [f"{axis.lower()}_{t}" for axis, t in wide.columns]
    for t in horizons:
        for axis in ("x", "y"):
            if f"{axis}_{t}" not in wide.columns:
                wide[f"{axis}_{t}"] = np.nan
    return wide.reset_index()


def load_parsed(parsed_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = json.loads((parsed_dir / "meta.json").read_text())
    return (meta,
            pd.read_parquet(parsed_dir / "rounds.parquet"),
            pd.read_parquet(parsed_dir / "players.parquet"),
            pd.read_parquet(parsed_dir / "trajectory.parquet"))


def build_demo_rows(parsed_dir: Path, zones: ZoneMapper) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta, rounds, players, trajectory = load_parsed(parsed_dir)

    df = players.rename(columns={"X": "spawn_x", "Y": "spawn_y"})
    df = df.merge(
        df.groupby(["round_number", "side"])
        .agg(team_money_avg=("money", "mean"), team_equip_avg=("equip_value", "mean"))
        .reset_index(),
        on=["round_number", "side"])
    df = df.merge(round_context(rounds, players), on=["round_number", "side"])
    df = df.merge(player_openings(players, trajectory, zones), on=["round_number", "steamid"], how="left")
    df["route"] = df["route"].fillna("unknown")
    df = df.merge(position_targets(trajectory, model_config()["labels"]["position_horizons_seconds"]),
                  on=["round_number", "steamid"], how="left")

    rosters = df.groupby("team")["steamid"].unique()
    names = df.groupby("team")["team_name"].agg(lambda s: next((n for n in s if n), ""))
    df["team_id"] = df["team"].map({t: team_id(r) for t, r in rosters.items()})
    df["team_name"] = df["team"].map(names)
    df["demo_id"] = meta["demo_id"]
    df["map_name"] = meta["map_name"]
    df = add_derived_features(df)

    team_rows = []
    rounds_by_num = rounds.set_index("round_number")
    for (rnd, side), g in df.groupby(["round_number", "side"]):
        r = rounds_by_num.loc[rnd]
        team_rows.append({
            "demo_id": meta["demo_id"], "map_name": meta["map_name"], "round_number": rnd, "side": side,
            "team_id": g["team_id"].iloc[0], "team_name": g["team_name"].iloc[0],
            "players": ", ".join(sorted(g["name"])),
            "buy_type_code": int(g["buy_type_code"].iloc[0]),
            "is_pistol": int(g["is_pistol"].iloc[0]),
            "won": int(r["winner"] == side),
            "plant_site": r["plant_site"] if isinstance(r["plant_site"], str) else None,
            "plant_seconds": r["plant_seconds"],
            "first_kill_side": r["first_kill_side"],
            "first_kill_place": r["first_kill_place"],
            "first_kill_seconds": r["first_kill_seconds"],
            "round_seconds": (r["end_tick"] - r["freeze_end_tick"]) / meta["tick_rate"]
                             if pd.notna(r["end_tick"]) else None,
            **team_strategy(g, r["plant_site"]),
        })
    return df, pd.DataFrame(team_rows)


def build_dataset(parsed_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dirs = sorted(p.parent for p in Path(parsed_root).glob("*/meta.json"))
    if not dirs:
        raise FileNotFoundError(f"No parsed demos found in {parsed_root}")

    # Site centres are learned per map from all of that map's demos
    by_map: dict[str, list[Path]] = {}
    for d in dirs:
        by_map.setdefault(json.loads((d / "meta.json").read_text())["map_name"], []).append(d)

    player_frames, team_frames = [], []
    for map_name, map_dirs in by_map.items():
        samples = pd.concat(pd.read_parquet(d / "trajectory.parquet") for d in map_dirs)
        zones = ZoneMapper(map_name, samples)
        for d in map_dirs:
            try:
                p, t = build_demo_rows(d, zones)
                player_frames.append(p)
                team_frames.append(t)
            except Exception as e:
                logger.error("Failed to build rows for %s: %s", d.name, e)
    return pd.concat(player_frames, ignore_index=True), pd.concat(team_frames, ignore_index=True)


def list_places(parsed_root: str | Path) -> pd.DataFrame:
    """Place names seen after freeze end, per map, with the zone they currently map to."""
    rows = []
    for meta_path in Path(parsed_root).glob("*/meta.json"):
        meta = json.loads(meta_path.read_text())
        traj = pd.read_parquet(meta_path.parent / "trajectory.parquet")
        zones = ZoneMapper(meta["map_name"], traj)
        for place, g in traj.groupby("last_place_name"):
            zone_counts = pd.Series([zones.zone(place, x, y) for x, y in zip(g["X"], g["Y"])]).value_counts()
            rows.append({"map_name": meta["map_name"], "place": place, "samples": len(g),
                         "zone": zone_counts.index[0] if not zone_counts.empty else "-"})
    out = pd.DataFrame(rows)
    return out.groupby(["map_name", "place", "zone"], as_index=False)["samples"].sum() \
              .sort_values(["map_name", "samples"], ascending=[True, False])


def main():
    ap = argparse.ArgumentParser(description="Build opening datasets from parsed demos")
    ap.add_argument("--parsed", default="data/parsed")
    ap.add_argument("--output", default="data/processed")
    ap.add_argument("--list-places", action="store_true", help="show place names per map and their zone")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.list_places:
        with pd.option_context("display.max_rows", None):
            print(list_places(args.parsed).to_string(index=False))
        return

    players, teams = build_dataset(args.parsed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    players.to_parquet(out / "players.parquet", index=False)
    teams.to_parquet(out / "teams.parquet", index=False)
    print(f"Saved {len(players)} player-rounds, {len(teams)} team-rounds "
          f"({players['demo_id'].nunique()} demos) to {out}")
    print(teams[teams["side"] == "T"].groupby("map_name")["strategy"].value_counts().to_string())


if __name__ == "__main__":
    main()
