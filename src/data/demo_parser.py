"""
CS2 Demo Parser - Extract round-start data from .dem files.

For every round it captures, at the moment freeze time ends:
  - rounds:     round_number, freeze_end_tick, end_tick, winner (T/CT)
  - players:    per-player snapshot (side, X/Y, money, equip value, place)
and after freeze end, sampled once per second until the round ends:
  - trajectory: per-player X/Y/place/is_alive at t = 1..N seconds
plus per round the bomb plant (site, time) and the opening kill.

Output per demo: data/parsed/<demo_id>/{rounds,players,trajectory}.parquet + meta.json
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from demoparser2 import DemoParser

from src.config import model_config

logger = logging.getLogger(__name__)

SNAPSHOT_FIELDS = ["X", "Y", "team_num", "current_equip_value", "balance", "last_place_name", "team_clan_name"]
TRAJECTORY_FIELDS = ["X", "Y", "last_place_name", "is_alive"]
TEAM_SIDES = {2: "T", 3: "CT"}


def place_site(place) -> str | None:
    if isinstance(place, str):
        if "BombsiteA" in place:
            return "A"
        if "BombsiteB" in place:
            return "B"
    return None


def normalize_winner(value) -> str | None:
    v = str(value).upper()
    if v in ("3", "CT"):
        return "CT"
    if v in ("2", "T", "TERRORIST"):
        return "T"
    return None


@dataclass
class ParsedDemo:
    meta: dict
    rounds: pd.DataFrame
    players: pd.DataFrame
    trajectory: pd.DataFrame

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.rounds.to_parquet(out_dir / "rounds.parquet", index=False)
        self.players.to_parquet(out_dir / "players.parquet", index=False)
        self.trajectory.to_parquet(out_dir / "trajectory.parquet", index=False)
        (out_dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
        return out_dir


class CS2DemoParser:
    """Parse a single CS2 .dem file into round-level tables."""

    def __init__(self, demo_path: str | Path, tick_rate: int | None = None, max_seconds: int | None = None):
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {self.demo_path}")
        labels = model_config()["labels"]
        self.tick_rate = tick_rate or labels["tick_rate"]
        self.max_seconds = max_seconds or labels["trajectory_max_seconds"]
        self.parser = DemoParser(str(self.demo_path))

    @property
    def demo_id(self) -> str:
        return self.demo_path.name.split(".")[0]

    def _ticks(self, fields: list[str], ticks: list[int]) -> pd.DataFrame:
        df = self.parser.parse_ticks(fields, ticks=[int(t) for t in ticks])
        return df[df["team_num"].isin(TEAM_SIDES)] if "team_num" in df.columns else df

    def parse_rounds(self) -> pd.DataFrame:
        freeze_end = self.parser.parse_event("round_freeze_end")
        if freeze_end.empty:
            raise ValueError("No round_freeze_end events in demo")
        freeze_ticks = sorted(freeze_end["tick"].unique())

        state = self.parser.parse_ticks(["total_rounds_played", "is_warmup_period"], ticks=freeze_ticks)
        state = state.groupby("tick", as_index=False).first()
        state = state[~state["is_warmup_period"].astype(bool)]
        state["round_number"] = state["total_rounds_played"].astype(int) + 1
        # A restarted round fires freeze_end again; the last one is the real one
        rounds = (
            state.sort_values("tick")
            .drop_duplicates("round_number", keep="last")
            .rename(columns={"tick": "freeze_end_tick"})[["round_number", "freeze_end_tick"]]
            .reset_index(drop=True)
        )

        ends = self.parser.parse_event("round_end")
        if not ends.empty:
            ends = ends[["tick", "winner"]].rename(columns={"tick": "end_tick"}).sort_values("end_tick")
            rounds = pd.merge_asof(
                rounds.sort_values("freeze_end_tick"), ends,
                left_on="freeze_end_tick", right_on="end_tick", direction="forward",
            )
            rounds["winner"] = rounds["winner"].map(normalize_winner)
            # Two rounds matched to the same end means a round had no end event
            rounds.loc[rounds.duplicated("end_tick", keep="last"), ["end_tick", "winner"]] = None
        else:
            rounds["end_tick"] = None
            rounds["winner"] = None
        rounds = self._attach_plants(rounds)
        return self._attach_opening_kills(rounds)

    def _round_of(self, events: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
        """Assign each event to the round whose live phase contains it."""
        ev = pd.merge_asof(
            events.sort_values("tick"), rounds[["round_number", "freeze_end_tick", "end_tick"]].sort_values("freeze_end_tick"),
            left_on="tick", right_on="freeze_end_tick", direction="backward",
        )
        ev = ev[ev["round_number"].notna() & (ev["end_tick"].isna() | (ev["tick"] <= ev["end_tick"]))]
        ev["t_seconds"] = (ev["tick"] - ev["freeze_end_tick"]) / self.tick_rate
        return ev

    def _attach_plants(self, rounds: pd.DataFrame) -> pd.DataFrame:
        rounds["plant_site"] = None
        rounds["plant_seconds"] = None
        plants = self.parser.parse_event("bomb_planted", player=["last_place_name"])
        if plants.empty:
            return rounds
        plants = self._round_of(plants, rounds).drop_duplicates("round_number", keep="first")
        plants["plant_site"] = plants["user_last_place_name"].map(place_site)
        info = plants.set_index("round_number")
        rounds["plant_site"] = rounds["round_number"].map(info["plant_site"])
        rounds["plant_seconds"] = rounds["round_number"].map(info["t_seconds"])
        return rounds

    def _attach_opening_kills(self, rounds: pd.DataFrame) -> pd.DataFrame:
        cols = ["first_kill_attacker", "first_kill_victim", "first_kill_side", "first_kill_place", "first_kill_seconds"]
        for c in cols:
            rounds[c] = None
        deaths = self.parser.parse_event("player_death", player=["last_place_name", "team_num"])
        if deaths.empty:
            return rounds
        deaths = deaths[deaths["attacker_steamid"].notna() & (deaths["attacker_steamid"] != deaths["user_steamid"])]
        first = self._round_of(deaths, rounds).drop_duplicates("round_number", keep="first").set_index("round_number")
        rounds["first_kill_attacker"] = rounds["round_number"].map(first["attacker_steamid"])
        rounds["first_kill_victim"] = rounds["round_number"].map(first["user_steamid"])
        rounds["first_kill_side"] = rounds["round_number"].map(first["attacker_team_num"].map(TEAM_SIDES))
        rounds["first_kill_place"] = rounds["round_number"].map(first["user_last_place_name"])
        rounds["first_kill_seconds"] = rounds["round_number"].map(first["t_seconds"])
        return rounds

    def parse_players(self, rounds: pd.DataFrame) -> pd.DataFrame:
        df = self._ticks(SNAPSHOT_FIELDS, rounds["freeze_end_tick"].tolist())
        df = df.merge(rounds[["freeze_end_tick", "round_number"]], left_on="tick", right_on="freeze_end_tick")
        df["side"] = df["team_num"].map(TEAM_SIDES)
        df = df.rename(columns={"current_equip_value": "equip_value", "balance": "money", "team_clan_name": "team_name"})
        return df[["round_number", "steamid", "name", "team_name", "side", "X", "Y",
                   "money", "equip_value", "last_place_name"]]

    def parse_trajectory(self, rounds: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for r in rounds.itertuples():
            for s in range(1, self.max_seconds + 1):
                tick = r.freeze_end_tick + s * self.tick_rate
                if pd.notna(r.end_tick) and tick > r.end_tick:
                    break
                rows.append((r.round_number, tick, s))
        samples = pd.DataFrame(rows, columns=["round_number", "tick", "t_seconds"])
        df = self._ticks(TRAJECTORY_FIELDS + ["team_num"], samples["tick"].tolist())
        df = df.merge(samples, on="tick")
        return df[["round_number", "t_seconds", "steamid", "X", "Y", "last_place_name", "is_alive"]]

    def parse(self) -> ParsedDemo:
        header = self.parser.parse_header()
        rounds = self.parse_rounds()
        players = self.parse_players(rounds)
        trajectory = self.parse_trajectory(rounds)
        meta = {
            "demo_id": self.demo_id,
            "source_file": str(self.demo_path),
            "map_name": header.get("map_name", "unknown"),
            "tick_rate": self.tick_rate,
            "num_rounds": int(len(rounds)),
        }
        logger.info("%s: %s, %d rounds, %d player-rounds",
                    self.demo_id, meta["map_name"], len(rounds), len(players))
        return ParsedDemo(meta, rounds, players, trajectory)


def parse_demo_directory(demo_dir: str | Path, output_dir: str | Path, overwrite: bool = False) -> list[Path]:
    """Parse every .dem in demo_dir. Already-parsed demos are skipped unless overwrite."""
    demo_dir, output_dir = Path(demo_dir), Path(output_dir)
    dem_files = sorted(demo_dir.glob("*.dem"))
    logger.info("Found %d demo files in %s", len(dem_files), demo_dir)

    outputs = []
    for dem_file in dem_files:
        out = output_dir / dem_file.stem
        if (out / "meta.json").exists() and not overwrite:
            logger.info("Skip (already parsed): %s", dem_file.name)
            outputs.append(out)
            continue
        try:
            outputs.append(CS2DemoParser(dem_file).parse().save(out))
        except Exception as e:
            logger.error("Failed to parse %s: %s", dem_file.name, e)

    logger.info("Parsed %d / %d demos", len(outputs), len(dem_files))
    return outputs


def main():
    ap = argparse.ArgumentParser(description="Parse CS2 demos into round-start tables")
    ap.add_argument("--input", default="data/raw/demos", help=".dem file or directory")
    ap.add_argument("--output", default="data/parsed")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    input_path = Path(args.input)
    if input_path.is_dir():
        parse_demo_directory(input_path, args.output, args.overwrite)
    else:
        CS2DemoParser(input_path).parse().save(Path(args.output) / input_path.stem)


if __name__ == "__main__":
    main()
