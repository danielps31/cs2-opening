"""XGBoost classifier for A/B/Mid route prediction at round start."""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
except ImportError:
    xgb = None


LABEL_MAP = {0: "A_site", 1: "B_site", 2: "Mid"}
CONFIG_PATH = Path(__file__).parents[2] / "configs" / "model_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["route_classifier"]


class RouteClassifier:
    """Predict player route (A / B / Mid) from round-start features."""

    def __init__(self):
        cfg = load_config()
        self.params = cfg["params"]
        self.model: xgb.XGBClassifier | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        if xgb is None:
            raise ImportError("xgboost required: pip install xgboost")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        preds = self.model.predict(X_test)
        report = classification_report(y_test, preds, target_names=list(LABEL_MAP.values()), output_dict=True)
        return report

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        if self.model:
            self.model.save_model(str(path))

    def load(self, path: str | Path) -> None:
        if xgb is None:
            raise ImportError("xgboost required")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))


if __name__ == "__main__":
    print("Route classifier — train with: python -m src.models.route_classifier --train")
