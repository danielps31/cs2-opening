"""Feature definitions shared by training and serving.

Both the dataset builder and the API go through `add_derived_features`, so the
model always sees the same columns in the same order.
"""

import numpy as np
import pandas as pd

from src.config import model_config

# Raw per-player-round fields (from demos, or sent to the API)
RAW_COLUMNS = [
    "side",            # "T" or "CT"
    "round_number",    # 1-based
    "score_team",      # own team's rounds won before this round
    "score_opponent",
    "prev_round_won",  # 1 if own team won the previous round, else 0
    "loss_bonus",      # $ own team gets if it loses this round
    "money",           # player's money at freeze end
    "equip_value",     # player's equipment value at freeze end
    "team_money_avg",
    "team_equip_avg",
    "spawn_x",
    "spawn_y",
]

FEATURE_COLUMNS = [
    "side_ct",
    "round_number",
    "is_pistol",
    "score_diff",
    "prev_round_won",
    "loss_bonus",
    "money",
    "equip_value",
    "buy_type_code",
    "team_money_avg",
    "team_equip_avg",
    "spawn_x",
    "spawn_y",
]

BUY_TYPES = {"eco": 0, "force": 1, "full": 2}


def classify_buy_type(equip_value: float) -> str:
    cfg = model_config()["features"]
    if equip_value < cfg["eco_threshold"]:
        return "eco"
    if equip_value < cfg["force_threshold"]:
        return "force"
    return "full"


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model features computed from RAW_COLUMNS. Returns a copy."""
    out = df.copy()
    out["side_ct"] = (out["side"] == "CT").astype(int)
    # Pistol rounds: first round of each regulation half (MR12)
    out["is_pistol"] = out["round_number"].isin([1, 13]).astype(int)
    out["score_diff"] = out["score_team"] - out["score_opponent"]
    out["buy_type_code"] = out["team_equip_avg"].map(lambda v: BUY_TYPES[classify_buy_type(v)])
    return out


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
