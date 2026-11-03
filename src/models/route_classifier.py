"""XGBoost classifier: which zone (A / B / Mid) a player takes in the opening."""

from pathlib import Path

import numpy as np
import xgboost as xgb

from src.config import model_config


class RouteClassifier:
    def __init__(self):
        cfg = model_config()["route_classifier"]
        self.params = cfg["params"]
        self.classes: list[str] = []
        self.model: xgb.XGBClassifier | None = None

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        order = model_config()["route_classifier"]["classes"]
        self.classes = [c for c in order if c in set(y)]
        index = {c: i for i, c in enumerate(self.classes)}
        val_mask = np.isin(y_val, self.classes)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X, np.array([index[c] for c in y]),
            eval_set=[(X_val[val_mask], np.array([index[c] for c in y_val[val_mask]]))],
            verbose=False,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(self.classes)[self.predict_proba(X).argmax(axis=1)]

    def save(self, path: str | Path) -> None:
        self.model.save_model(str(path))

    def load(self, path: str | Path, classes: list[str]) -> None:
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        self.classes = classes
