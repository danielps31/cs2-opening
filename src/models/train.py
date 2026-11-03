"""Train one route classifier + position predictor per map.

    python -m src.models.train [--map de_dust2]

Splits by demo when there are enough demos (rounds of one match never land in
both train and test), otherwise by round. Writes models/<map>/ with a meta.json
holding the metrics, including a majority-class baseline to compare against.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit

from src.config import model_config
from src.features.schema import FEATURE_COLUMNS, feature_matrix
from src.models.position_predictor import PositionPredictor
from src.models.route_classifier import RouteClassifier


def split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = model_config()["training"]
    groups = df["demo_id"] if df["demo_id"].nunique() >= 5 else df["demo_id"] + "_" + df["round_number"].astype(str)
    holdout = cfg["val_split"] + cfg["test_split"]
    train_idx, rest_idx = next(GroupShuffleSplit(1, test_size=holdout, random_state=seed).split(df, groups=groups))
    rest = df.iloc[rest_idx]
    val_idx, test_idx = next(GroupShuffleSplit(1, test_size=cfg["test_split"] / holdout, random_state=seed)
                             .split(rest, groups=groups.iloc[rest_idx]))
    return df.iloc[train_idx], rest.iloc[val_idx], rest.iloc[test_idx]


def train_map(df: pd.DataFrame, map_name: str, out_dir: Path) -> dict:
    cfg = model_config()
    seed = cfg["training"]["random_seed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"map_name": map_name, "features": FEATURE_COLUMNS, "rows": len(df), "demos": int(df["demo_id"].nunique())}

    labelled = df[df["route"].isin(cfg["route_classifier"]["classes"])]
    tr, va, te = split(labelled, seed)
    clf = RouteClassifier()
    clf.train(feature_matrix(tr), tr["route"].to_numpy(), feature_matrix(va), va["route"].to_numpy())
    clf.save(out_dir / "route_classifier.json")
    pred = clf.predict(feature_matrix(te))
    majority = tr["route"].mode().iloc[0]
    meta["route"] = {
        "classes": clf.classes,
        "test_rows": len(te),
        "accuracy": round(accuracy_score(te["route"], pred), 3),
        "baseline_accuracy": round(accuracy_score(te["route"], [majority] * len(te)), 3),
        "report": classification_report(te["route"], pred, output_dict=True, zero_division=0),
    }

    horizons = cfg["labels"]["position_horizons_seconds"]
    target_cols = [f"{a}_{t}" for t in horizons for a in ("x", "y")]
    pos = df.dropna(subset=target_cols)
    tr, va, te = split(pos, seed)
    model = PositionPredictor(len(FEATURE_COLUMNS), horizons)
    model.train(feature_matrix(tr), tr[target_cols].to_numpy(np.float32),
                feature_matrix(va), va[target_cols].to_numpy(np.float32))
    model.save(out_dir / "position_predictor.pt")
    err = model.predict(feature_matrix(te)) - te[target_cols].to_numpy(np.float32)
    dist = np.sqrt(err[:, 0::2] ** 2 + err[:, 1::2] ** 2)
    meta["position"] = {
        "horizons": horizons,
        "test_rows": len(te),
        "mean_error_units": {str(t): round(float(dist[:, i].mean()), 1) for i, t in enumerate(horizons)},
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description="Train route + position models per map")
    ap.add_argument("--dataset", default="data/processed/players.parquet")
    ap.add_argument("--map", default=None)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    min_rows = model_config()["training"]["min_rows_per_map"]
    maps = [args.map] if args.map else sorted(df["map_name"].unique())
    for map_name in maps:
        mdf = df[df["map_name"] == map_name]
        if len(mdf) < min_rows:
            print(f"{map_name}: skip, only {len(mdf)} rows (need {min_rows})")
            continue
        meta = train_map(mdf, map_name, Path(args.out) / map_name)
        r, p = meta["route"], meta["position"]
        print(f"{map_name}: route acc {r['accuracy']} (baseline {r['baseline_accuracy']}, {r['test_rows']} test rows), "
              f"position error {p['mean_error_units']} units")


if __name__ == "__main__":
    main()
