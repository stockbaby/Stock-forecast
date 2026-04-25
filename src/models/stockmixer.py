from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.models.deep_sequence import SequenceDatasetBundle


@dataclass
class StockMixerTrainConfig:
    hidden_dim: int = 128
    mixer_dim: int = 256
    temporal_dim: int = 64
    dropout: float = 0.1
    batch_size: int = 512
    epochs: int = 10
    learning_rate: float = 1e-3


def train_stockmixer_regressor(
    dataset: SequenceDatasetBundle,
    config: StockMixerTrainConfig,
) -> tuple[object, pd.DataFrame]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is required for the StockMixer baseline. Install PyTorch before running this script."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = dataset.x_train.shape[-1]
    seq_len = dataset.x_train.shape[1]

    class MlpBlock(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, in_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class MixerLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.indicator_norm = nn.LayerNorm(config.hidden_dim)
            self.indicator_mlp = MlpBlock(config.hidden_dim, config.mixer_dim, config.dropout)

            self.temporal_norm = nn.LayerNorm(config.hidden_dim)
            self.temporal_proj = nn.Sequential(
                nn.Linear(seq_len, config.temporal_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.temporal_dim, seq_len),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.indicator_mlp(self.indicator_norm(x))

            y = self.temporal_norm(x).transpose(1, 2)
            y = self.temporal_proj(y).transpose(1, 2)
            x = x + y
            return x

    class StockMixerRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, config.hidden_dim)
            self.mixer1 = MixerLayer()
            self.mixer2 = MixerLayer()
            self.time_pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_proj(x)
            x = self.mixer1(x)
            x = self.mixer2(x)
            pooled = self.time_pool(x.transpose(1, 2)).squeeze(-1)
            return self.head(pooled).squeeze(-1)

    model = StockMixerRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(dataset.x_train), torch.from_numpy(dataset.y_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model.train()
    for _ in range(config.epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_valid = torch.from_numpy(dataset.x_valid).to(device)
        preds = model(x_valid).detach().cpu().numpy()

    valid_pred_df = dataset.valid_meta.copy()
    valid_pred_df["score"] = preds
    valid_pred_df["label"] = dataset.y_valid
    return model, valid_pred_df
