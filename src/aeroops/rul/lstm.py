"""Small CPU-friendly LSTM regressor for RUL."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RULLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def train_lstm(X, y, epochs: int = 30, lr: float = 1e-3, batch_size: int = 256, seed: int = 42, log=None, y_scale: float = 125.0):
    """Targets are divided by y_scale during training (unscaled RUL targets stall the LSTM at the mean)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RULLSTM(X.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y / y_scale)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if log:
            log(ep, total / n * y_scale**2)
    model.y_scale = y_scale
    return model


@torch.no_grad()
def predict_lstm(model, X) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(X)).numpy() * getattr(model, "y_scale", 1.0)
