"""FastAPI prediction endpoint: route probabilities and positions for one player at round start."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features.schema import add_derived_features, feature_matrix
from src.models.position_predictor import PositionPredictor
from src.models.route_classifier import RouteClassifier

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
MODELS: dict[str, dict] = {}


def load_models(model_dir: Path = MODEL_DIR) -> None:
    MODELS.clear()
    for meta_path in model_dir.glob("*/meta.json"):
        meta = json.loads(meta_path.read_text())
        clf = RouteClassifier()
        clf.load(meta_path.parent / "route_classifier.json", meta["route"]["classes"])
        MODELS[meta["map_name"]] = {
            "meta": meta, "route": clf,
            "position": PositionPredictor.load(meta_path.parent / "position_predictor.pt"),
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(title="CS2 Opening Predictor", version="0.2.0", lifespan=lifespan)


class RoundState(BaseModel):
    """One player at freeze end. Team averages default to the player's own values."""
    map_name: str
    side: str  # "T" or "CT"
    round_number: int
    score_team: int = 0
    score_opponent: int = 0
    prev_round_won: bool = False
    loss_bonus: int = 1400
    money: float
    equip_value: float
    team_money_avg: float | None = None
    team_equip_avg: float | None = None
    spawn_x: float
    spawn_y: float


class Prediction(BaseModel):
    route: str
    route_probabilities: dict[str, float]
    predicted_positions: list[dict[str, float]]  # [{t_seconds, x, y}]


@app.get("/health")
async def health():
    return {"status": "ok", "maps": sorted(MODELS)}


@app.post("/predict", response_model=Prediction)
async def predict(state: RoundState):
    models = MODELS.get(state.map_name)
    if models is None:
        raise HTTPException(404, f"No model for {state.map_name}. Loaded: {sorted(MODELS)}")

    row = state.model_dump()
    row["prev_round_won"] = int(row["prev_round_won"])
    row["team_money_avg"] = row["team_money_avg"] if row["team_money_avg"] is not None else row["money"]
    row["team_equip_avg"] = row["team_equip_avg"] if row["team_equip_avg"] is not None else row["equip_value"]
    X = feature_matrix(add_derived_features(pd.DataFrame([row])))

    clf = models["route"]
    probs = clf.predict_proba(X)[0]
    positions = models["position"].predict(X)[0]
    horizons = models["position"].horizons
    return Prediction(
        route=clf.classes[int(probs.argmax())],
        route_probabilities={c: round(float(p), 3) for c, p in zip(clf.classes, probs)},
        predicted_positions=[
            {"t_seconds": float(t), "x": float(positions[2 * i]), "y": float(positions[2 * i + 1])}
            for i, t in enumerate(horizons)
        ],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("API_HOST", "127.0.0.1"), port=int(os.environ.get("API_PORT", 8000)))
