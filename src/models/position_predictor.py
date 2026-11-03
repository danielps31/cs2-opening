"""MLP predicting a player's x,y at each horizon (e.g. 5/10/15 s after freeze end)."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import model_config


class PositionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PositionPredictor:
    """Inputs and targets are standardised; the stats travel with the checkpoint."""

    def __init__(self, input_dim: int, horizons: list[int] | None = None):
        cfg = model_config()
        self.params = cfg["position_predictor"]["params"]
        self.horizons = horizons or cfg["labels"]["position_horizons_seconds"]
        self.input_dim = input_dim
        self.model = PositionMLP(input_dim, self.params["hidden_layers"], 2 * len(self.horizons), self.params["dropout"])
        self.stats: dict[str, np.ndarray] = {}

    def _norm(self, X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((X - self.stats["x_mean"]) / self.stats["x_std"]).astype(np.float32))

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Train with early stopping. Returns the best validation RMSE in world units."""
        self.stats = {"x_mean": X.mean(0), "x_std": X.std(0) + 1e-6, "y_mean": y.mean(0), "y_std": y.std(0) + 1e-6}
        yt = torch.from_numpy(((y - self.stats["y_mean"]) / self.stats["y_std"]).astype(np.float32))
        loader = DataLoader(TensorDataset(self._norm(X), yt), batch_size=self.params["batch_size"], shuffle=True)
        Xv = self._norm(X_val)
        yv = torch.from_numpy(((y_val - self.stats["y_mean"]) / self.stats["y_std"]).astype(np.float32))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])
        loss_fn = nn.MSELoss()
        best, best_state, stale = float("inf"), None, 0
        for _ in range(self.params["epochs"]):
            self.model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss_fn(self.model(xb), yb).backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val = loss_fn(self.model(Xv), yv).item()
            if val < best:
                best, best_state, stale = val, {k: v.clone() for k, v in self.model.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= self.params["patience"]:
                    break
        self.model.load_state_dict(best_state)
        pred = self.predict(X_val)
        return float(np.sqrt(((pred - y_val) ** 2).mean()))

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            out = self.model(self._norm(X)).numpy()
        return out * self.stats["y_std"] + self.stats["y_mean"]

    def save(self, path: str | Path) -> None:
        torch.save({"state_dict": self.model.state_dict(), "stats": self.stats,
                    "horizons": self.horizons, "input_dim": self.input_dim}, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "PositionPredictor":
        ckpt = torch.load(str(path), weights_only=False)
        obj = cls(ckpt["input_dim"], ckpt["horizons"])
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.stats = ckpt["stats"]
        return obj
