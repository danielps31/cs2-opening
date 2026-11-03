"""Feature engineering for CS2 round-start movement prediction."""

import pandas as pd
import numpy as np
from pathlib import Path


# Buy type thresholds (approximate CS2 values)
ECO_THRESHOLD = 1500
FORCE_THRESHOLD = 3500


def classify_buy_type(equipment_value: float) -> str:
    if equipment_value < ECO_THRESHOLD:
        return "eco"
    elif equipment_value < FORCE_THRESHOLD:
        return "force"
    return "full"


def compute_loss_bonus(consecutive_losses: int) -> int:
    """CS2 loss bonus: $1400 base + $500 per consecutive loss, max $3400."""
    return min(1400 + 500 * consecutive_losses, 3400)


def extract_features(
    positions_df: pd.DataFrame,
    rounds_df: pd.DataFrame,
    players_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature matrix from parsed demo data.

    Features per player per round:
    - equipment_value: total equipment value at freeze time end
    - freeze_money: money at end of freeze time
    - loss_bonus: current loss bonus amount
    - team_economy_avg: average team economy
    - side: T (0) or CT (1)
    - map_name: one-hot encoded map
    - round_number: current round (1-30+)
    - score_diff: team score - opponent score
    - buy_type: eco/force/full (encoded)
    - prev_round_won: whether team won previous round
    - spawn_x, spawn_y: initial spawn position
    """
    features = []

    for _, round_row in rounds_df.iterrows():
        round_num = round_row.get("round", 0)
        round_tick = round_row.get("tick", 0)

        # Get positions at round start
        start_positions = positions_df[positions_df["tick"] == round_tick]

        for _, pos in start_positions.iterrows():
            feat = {
                "round_number": round_num,
                "spawn_x": pos.get("X", 0.0),
                "spawn_y": pos.get("Y", 0.0),
                "side": 1 if pos.get("team", "") == "CT" else 0,
                "equipment_value": pos.get("equipment_value", 0),
                "is_alive": pos.get("is_alive", True),
                "player_id": pos.get("steamid", ""),
            }
            feat["buy_type_encoded"] = {"eco": 0, "force": 1, "full": 2}.get(
                classify_buy_type(feat["equipment_value"]), 0
            )
            features.append(feat)

    return pd.DataFrame(features)


def build_labels(
    positions_df: pd.DataFrame,
    round_start_tick: int,
    tick_rate: int = 64,
    horizon_seconds: float = 15.0,
) -> pd.DataFrame:
    """Build route labels from player positions at horizon time.

    Route classification based on position relative to map geometry:
    - A_site, B_site, Mid (map-specific thresholds needed)
    """
    horizon_tick = round_start_tick + int(horizon_seconds * tick_rate)
    at_horizon = positions_df[positions_df["tick"] == horizon_tick]

    labels = at_horizon[["steamid", "X", "Y"]].copy()
    labels.rename(columns={"X": "target_x", "Y": "target_y"}, inplace=True)
    # Route label would be assigned based on map-specific geometry
    labels["route"] = "unknown"
    return labels


if __name__ == "__main__":
    print("Feature engineer — run after demo parsing.")
    print("Usage: python -m src.features.engineer")
