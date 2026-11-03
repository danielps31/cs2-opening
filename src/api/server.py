"""FastAPI prediction endpoint for CS2 movement predictions."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="CS2 Movement Predictor", version="0.1.0")


class RoundState(BaseModel):
    """Input: current round state for prediction."""
    map_name: str = "de_dust2"
    round_number: int
    side: str  # "T" or "CT"
    score_team: int
    score_opponent: int
    equipment_value: float
    freeze_money: float
    loss_bonus: int
    buy_type: str  # "eco", "force", "full"
    spawn_x: float
    spawn_y: float


class Prediction(BaseModel):
    """Output: predicted route and positions."""
    route: str  # "A_site", "B_site", "Mid"
    route_probabilities: dict[str, float]
    predicted_positions: list[dict[str, float]]  # [{x, y, t_seconds}]


# Models loaded at startup
route_model = None
position_model = None


@app.on_event("startup")
async def load_models():
    global route_model, position_model
    # TODO: Load trained models from disk
    pass


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": route_model is not None}


@app.post("/predict", response_model=Prediction)
async def predict(state: RoundState):
    if route_model is None:
        raise HTTPException(503, "Models not loaded")

    # Build feature vector from round state
    features = np.array([[
        state.round_number,
        1 if state.side == "CT" else 0,
        state.score_team - state.score_opponent,
        state.equipment_value,
        state.freeze_money,
        state.loss_bonus,
        {"eco": 0, "force": 1, "full": 2}.get(state.buy_type, 0),
        state.spawn_x,
        state.spawn_y,
    ]])

    route_probs = route_model.predict_proba(features)[0]
    route_idx = int(np.argmax(route_probs))
    route_labels = ["A_site", "B_site", "Mid"]

    positions = position_model.predict(features)

    return Prediction(
        route=route_labels[route_idx],
        route_probabilities=dict(zip(route_labels, route_probs.tolist())),
        predicted_positions=[
            {"x": float(positions[0][0]), "y": float(positions[0][1]), "t_seconds": 15.0}
        ],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
