"""Lightweight MLP for predicting player x,y coordinates at future timesteps."""

import yaml
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "model_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["position_predictor"]


class PositionMLP(nn.Module if nn else object):
    """Multi-layer perceptron for x,y coordinate prediction."""

    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float = 0.3):
        if nn is None:
            raise ImportError("torch required: pip install torch")
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))  # output: x, y
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PositionPredictor:
    """Train and predict player positions using PositionMLP."""

    def __init__(self, input_dim: int = 10):
        cfg = load_config()
        self.config = cfg
        self.model = PositionMLP(
            input_dim=input_dim,
            hidden_layers=cfg["params"]["hidden_layers"],
            dropout=cfg["params"]["dropout"],
        ) if nn else None
        self.device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")

    def train(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        if torch is None:
            raise ImportError("torch required")

        self.model.to(self.device)
        self.model.train()

        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config["params"]["batch_size"],
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["params"]["learning_rate"],
        )
        criterion = nn.MSELoss()
        losses = []

        for epoch in range(self.config["params"]["epochs"]):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            return self.model(tensor).cpu().numpy()

    def save(self, path: str | Path) -> None:
        if self.model:
            torch.save(self.model.state_dict(), str(path))

    def load(self, path: str | Path, input_dim: int = 10) -> None:
        if torch is None:
            raise ImportError("torch required")
        cfg = load_config()
        self.model = PositionMLP(input_dim, cfg["params"]["hidden_layers"], cfg["params"]["dropout"])
        self.model.load_state_dict(torch.load(str(path), weights_only=True))


if __name__ == "__main__":
    print("Position predictor — MLP for x,y coordinate prediction")
