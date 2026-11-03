"""
CS2 Demo Parser - Extract player movement data from .dem files.

Uses demoparser2 to parse Counter-Strike 2 demo files and extract
per-tick player positions, economy data, and round information.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

logger = logging.getLogger(__name__)

# Fields we pull from demoparser2 per tick
TICK_FIELDS = [
    "X", "Y", "Z",
    "pitch", "yaw",
    "health", "armor_value",
    "is_alive",
    "team_num",          # 2=T, 3=CT
    "active_weapon_name",
    "current_equip_value",
    "total_rounds_played",
]

PLAYER_INFO_FIELDS = [
    "steamid",
    "name",
    "team_num",
]


class CS2DemoParser:
    """Parse a single CS2 .dem file into structured DataFrames."""

    def __init__(self, demo_path: str | Path):
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {self.demo_path}")
        self.parser = DemoParser(str(self.demo_path))

    def parse_header(self) -> dict:
        """Extract demo header info (map name, duration, etc.)."""
        header = self.parser.parse_header()
        return header

    def parse_ticks(self, tick_fields: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Parse per-tick player data.

        Returns DataFrame with columns:
            tick, steamid, X, Y, Z, pitch, yaw, health, armor_value,
            is_alive, team_num, active_weapon_name, current_equip_value,
            total_rounds_played
        """
        fields = tick_fields or TICK_FIELDS
        logger.info("Parsing ticks from %s with fields: %s", self.demo_path.name, fields)

        df = self.parser.parse_ticks(fields)

        # demoparser2 returns a DataFrame with 'tick' and 'steamid' as columns
        logger.info("Parsed %d tick rows from %s", len(df), self.demo_path.name)
        return df

    def parse_events(self, event_name: str) -> pd.DataFrame:
        """Parse specific game events (round_start, player_death, etc.)."""
        df = self.parser.parse_event(event_name)
        logger.info("Parsed %d '%s' events", len(df), event_name)
        return df

    def parse_round_starts(self) -> pd.DataFrame:
        """Get round start ticks and associated data."""
        return self.parse_events("round_start")

    def parse_round_ends(self) -> pd.DataFrame:
        """Get round end ticks and results."""
        return self.parse_events("round_end")

    def parse_player_deaths(self) -> pd.DataFrame:
        """Get player death events."""
        return self.parse_events("player_death")

    def extract_economy(self) -> pd.DataFrame:
        """
        Extract per-round economy data for each player.

        Returns DataFrame with: steamid, round_number, start_money,
        equip_value, team_num
        """
        # Parse round freeze end events to capture economy at round start
        try:
            freeze_end = self.parse_events("round_freeze_end")
        except Exception:
            logger.warning("No round_freeze_end events; falling back to round_start")
            freeze_end = self.parse_round_starts()

        # Get tick data at those specific ticks to capture economy snapshot
        eco_fields = ["current_equip_value", "team_num", "total_rounds_played"]
        ticks = self.parser.parse_ticks(eco_fields)

        if "total_rounds_played" not in ticks.columns:
            ticks["total_rounds_played"] = 0

        # Group by round and player, take first tick per round as economy snapshot
        economy = (
            ticks.groupby(["steamid", "total_rounds_played"])
            .agg(
                equip_value=("current_equip_value", "first"),
                team_num=("team_num", "first"),
            )
            .reset_index()
            .rename(columns={"total_rounds_played": "round_number"})
        )

        return economy

    def extract_movement_data(self) -> pd.DataFrame:
        """
        Full extraction pipeline: ticks + header merged into one DataFrame.

        Adds map_name column from header and computes velocity.
        """
        header = self.parse_header()
        ticks = self.parse_ticks()

        # Add map name
        map_name = header.get("map_name", "unknown")
        ticks["map_name"] = map_name

        # Sort for velocity computation
        ticks = ticks.sort_values(["steamid", "tick"]).reset_index(drop=True)

        # Compute velocity (units/tick) per player
        grouped = ticks.groupby("steamid")
        ticks["dx"] = grouped["X"].diff().fillna(0.0)
        ticks["dy"] = grouped["Y"].diff().fillna(0.0)
        ticks["dz"] = grouped["Z"].diff().fillna(0.0)
        ticks["velocity"] = np.sqrt(ticks["dx"] ** 2 + ticks["dy"] ** 2 + ticks["dz"] ** 2)

        return ticks

    def to_parquet(self, output_path: str | Path, ticks_df: Optional[pd.DataFrame] = None):
        """Save parsed data to parquet format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if ticks_df is None:
            ticks_df = self.extract_movement_data()

        ticks_df.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info("Saved %d rows to %s", len(ticks_df), output_path)
        return output_path


def parse_demo_directory(
    demo_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.dem",
) -> list[Path]:
    """
    Batch-parse all .dem files in a directory to parquet.

    Returns list of output parquet file paths.
    """
    demo_dir = Path(demo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = []
    dem_files = sorted(demo_dir.glob(pattern))
    logger.info("Found %d demo files in %s", len(dem_files), demo_dir)

    for dem_file in dem_files:
        try:
            parser = CS2DemoParser(dem_file)
            out_path = output_dir / f"{dem_file.stem}.parquet"
            parser.to_parquet(out_path)
            output_files.append(out_path)
        except Exception as e:
            logger.error("Failed to parse %s: %s", dem_file.name, e)

    logger.info("Successfully parsed %d / %d demos", len(output_files), len(dem_files))
    return output_files


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python demo_parser.py <demo_file_or_dir> [output_dir]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/parsed")

    if input_path.is_dir():
        parse_demo_directory(input_path, output_dir)
    else:
        parser = CS2DemoParser(input_path)
        parser.to_parquet(output_dir / f"{input_path.stem}.parquet")
