"""Build the opening scouting report: one self-contained HTML page.

    python -m src.report.build_report [--me <steamid>] [--output reports/opening_report.html]

The report opens on one player (--me, default: the steamid present in the most
demos); any player can be picked by Steam ID, and only their matches are shown.
"""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

from src.config import map_radar
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


def radar_for(map_name: str, radar_dir: Path) -> dict | None:
    """Radar image (as a data URL) + world-to-pixel transform, extracting from the game if needed."""
    png = radar_dir / f"{map_name}.png"
    overviews_path = radar_dir / "overviews.json"
    if not png.exists():
        try:
            from src.data.radar import extract
            extract([map_name], radar_dir)
        except Exception as e:
            print(f"{map_name}: no radar image ({e})")
    if not png.exists():
        return None
    overviews = json.loads(overviews_path.read_text()) if overviews_path.exists() else {}
    transform = overviews.get(map_name) or map_radar(map_name)
    if not transform:
        return None
    img = Image.open(png).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=80)
    # Crop the view to the drawn part of the radar (some maps only fill the middle)
    x0, y0, x1, y1 = img.getchannel("A").getbbox() or (0, 0, *img.size)
    side = max(x1 - x0, y1 - y0) + 32
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return {"image": "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode(),
            "size": img.size[0], "view": [round(cx - side / 2), round(cy - side / 2), side], **transform}


def callouts(traj: pd.DataFrame) -> list[list]:
    """[place, x, y] at the median position of each nav place."""
    named = traj[traj["last_place_name"].fillna("") != ""]
    med = named.groupby("last_place_name").agg(X=("X", "median"), Y=("Y", "median"), n=("X", "size"))
    med = med[med["n"] >= 5].sort_values("n", ascending=False)
    return [[place, round(r.X), round(r.Y)] for place, r in med.iterrows()]


def weapon_key(weapon: str) -> str:
    """Kill-feed weapon name -> icon name."""
    w = str(weapon)
    if w.startswith(("knife", "bayonet")):
        return "knife"
    return w.removesuffix("_off")


def combat_rows(parsed: Path, match_index: dict, player_index: dict) -> tuple[list, list]:
    """Compact hit and kill rows: [match, round, attacker, victim, attacker side, ...]."""
    hits, kills = [], []
    for meta_path in sorted(parsed.glob("*/meta.json")):
        demo = meta_path.parent
        if not (demo / "hits.parquet").exists():
            continue
        demo_id = json.loads(meta_path.read_text())["demo_id"]
        m = match_index.get(demo_id)
        if m is None:
            continue
        for name, out, extra in (("hits", hits, ["hitgroup", "dmg_health"]), ("kills", kills, ["weapon", "headshot"])):
            df = pd.read_parquet(demo / f"{name}.parquet")
            for r in df.itertuples(index=False):
                att, vic = str(r.attacker_steamid), str(r.user_steamid)
                if att not in player_index or vic not in player_index:
                    continue
                row = [m, int(r.round_number), player_index[att], player_index[vic], r.attacker_side]
                if name == "hits":
                    row += [r.hitgroup, int(r.dmg_health)]
                else:
                    row += [weapon_key(r.weapon), int(bool(r.headshot))]
                out.append(row)
    return hits, kills


def icon_uris(weapons: set[str], icon_dir: Path) -> dict[str, str]:
    out = {}
    for w in weapons:
        svg = icon_dir / f"{w}.svg"
        if svg.exists():
            out[w] = "data:image/svg+xml;base64," + base64.b64encode(svg.read_bytes()).decode()
    return out


def match_list(teams: pd.DataFrame) -> list[dict]:
    """One entry per demo with rounds won per team, so the page can show any player's score."""
    out = []
    for demo_id, g in teams.groupby("demo_id", sort=False):
        out.append({
            "demo_id": demo_id,
            "map_name": g["map_name"].iloc[0],
            "rounds": int(g["round_number"].max()),
            "team_wins": {t: int(w) for t, w in g.groupby("team_id")["won"].sum().items()},
        })
    return out


def build_payload(processed: Path, parsed: Path, me: str | None, radar_dir: Path) -> dict:
    players = pd.read_parquet(processed / "players.parquet")
    teams = pd.read_parquet(processed / "teams.parquet")
    # steamid64 exceeds JS number precision, so it travels as a string
    players["steamid"] = players["steamid"].astype(str)
    me = me or detect_me(players)
    me_name = players.loc[players["steamid"] == me, "name"].iloc[0] if (players["steamid"] == me).any() else me

    players["buy"] = [buy_label(p, b) for p, b in zip(players["is_pistol"], players["buy_type_code"])]
    teams["buy"] = [buy_label(p, b) for p, b in zip(teams["is_pistol"], teams["buy_type_code"])]
    teams["team_label"] = [n or p for n, p in zip(teams["team_name"], teams["players"])]

    matches = match_list(teams)
    match_index = {m["demo_id"]: i for i, m in enumerate(matches)}

    # Trajectory dots (every 2 s) for the heatmap and per-round paths
    roster = players.drop_duplicates("steamid")[["steamid", "name"]].reset_index(drop=True)
    player_index = {sid: i for i, sid in enumerate(roster["steamid"])}
    dots, centres, radars, places = [], {}, {}, {}
    route_of = players.set_index(["demo_id", "round_number", "steamid"])[["route", "side"]]
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
        places[map_name] = callouts(traj)
        radars[map_name] = radar_for(map_name, radar_dir)
        traj = traj.join(route_of, on=["demo_id", "round_number", "steamid"], how="inner")
        traj["map_name"] = map_name
        traj["player"] = traj["steamid"].map(player_index)
        traj["match"] = traj["demo_id"].map(match_index)
        dots.append(traj[["map_name", "side", "route", "t_seconds", "X", "Y", "player", "match", "round_number"]])
    hits, kills = combat_rows(parsed, match_index, player_index)
    icon_dir = radar_dir.parent / "icons"
    if not icon_dir.exists():
        try:
            from src.data.radar import extract_icons
            extract_icons(icon_dir)
        except Exception as e:
            print(f"No weapon icons ({e})")

    dots_df = pd.concat(dots)
    dots_df["X"] = dots_df["X"].round().astype(int)
    dots_df["Y"] = dots_df["Y"].round().astype(int)

    team_cols = ["demo_id", "map_name", "round_number", "side", "team_id", "team_label", "buy",
                 "strategy", "n_A", "n_B", "n_Mid", "rushers", "rush_site", "plant_site", "won",
                 "first_kill_side", "first_kill_place", "first_kill_seconds", "round_seconds"]
    player_cols = ["demo_id", "map_name", "round_number", "side", "steamid", "name", "team_id", "buy",
                   "route", "rush", "site_arrival_seconds", "opening_line"]
    return {
        "me": {"steamid": me, "name": me_name},
        "summary": {
            "demos": int(players["demo_id"].nunique()),
            "rounds": int(teams[["demo_id", "round_number"]].drop_duplicates().shape[0]),
            "players": int(players["steamid"].nunique()),
        },
        "matches": matches,
        "teams": records(teams[team_cols]),
        "players": records(players[player_cols]),
        "dots": dots_df.values.tolist(),
        "dot_columns": list(dots_df.columns),
        "centres": centres,
        "radars": radars,
        "places": places,
        "roster": roster.values.tolist(),
        "hits": hits,
        "kills": kills,
        "icons": icon_uris({k[5] for k in kills}, icon_dir),
    }


def main():
    ap = argparse.ArgumentParser(description="Build the HTML opening report")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--parsed", default="data/parsed")
    ap.add_argument("--me", default=None, help="your steamid64 (default: auto-detect)")
    ap.add_argument("--radar", default="data/radar", help="radar images (auto-extracted from CS2 if missing)")
    ap.add_argument("--output", default="reports/opening_report.html")
    args = ap.parse_args()

    payload = build_payload(Path(args.processed), Path(args.parsed), args.me, Path(args.radar))
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, default=clean))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Report: {out.resolve()}  (me = {payload['me']['name']})")


if __name__ == "__main__":
    main()
