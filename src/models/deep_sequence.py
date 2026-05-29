from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SequenceDatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    train_meta: pd.DataFrame
    x_valid: np.ndarray
    y_valid: np.ndarray
    valid_meta: pd.DataFrame
    feature_columns: list[str]


MARKET_CONTEXT_COLUMNS = [
    "regime_trend",
    "regime_vol_ratio",
    "regime_drawdown",
    "regime_score",
    "regime_is_trending",
    "regime_is_high_vol",
    "index_ret_1",
    "index_ret_5",
    "index_ret_10",
    "index_ret_20",
    "index_volatility_5",
    "index_volatility_20",
    "index_drawdown_20",
    "ret_5",
    "ret_20",
    "volatility_20",
    "volume_ratio_5",
    "volume_ratio_20",
    "amount_ratio_5",
    "amount_ratio_20",
    "volume_breakout_5",
    "volume_breakout_20",
    "industry_id",
    "industry_collective_momentum",
    "industry_collective_volume_confirm",
]


def _meta_context(row: pd.Series) -> dict[str, Any]:
    return {col: row[col] for col in MARKET_CONTEXT_COLUMNS if col in row.index}


def build_lstm_sequences(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    lookback: int = 20,
) -> SequenceDatasetBundle:
    x_train, y_train, train_meta = _build_sequences_for_frame(train_df, feature_columns, label_column, lookback)
    x_valid, y_valid, valid_meta = _build_sequences_for_frame(valid_df, feature_columns, label_column, lookback)
    return SequenceDatasetBundle(
        x_train=x_train,
        y_train=y_train,
        train_meta=train_meta,
        x_valid=x_valid,
        y_valid=y_valid,
        valid_meta=valid_meta,
        feature_columns=feature_columns,
    )


def build_prediction_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    lookback: int = 20,
    target_dates: list[pd.Timestamp] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    sequences: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []

    target_set = None
    if target_dates is not None:
        target_set = {pd.Timestamp(date).normalize() for date in target_dates}

    df = df.sort_values(["stock_id", "date"]).copy()
    for stock_id, group in df.groupby("stock_id"):
        group = group.reset_index(drop=True)
        x = group[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        dates = pd.to_datetime(group["date"])

        for idx in range(lookback - 1, len(group)):
            date = pd.Timestamp(dates.iloc[idx]).normalize()
            if target_set is not None and date not in target_set:
                continue
            seq = x[idx - lookback + 1 : idx + 1]
            sequences.append(seq)
            metas.append({"stock_id": str(stock_id), "date": dates.iloc[idx], **_meta_context(group.iloc[idx])})

    if not sequences:
        return (
            np.empty((0, lookback, len(feature_columns)), dtype=np.float32),
            pd.DataFrame(columns=["stock_id", "date"]),
        )

    return np.stack(sequences), pd.DataFrame(metas)


def _build_sequences_for_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    metas: list[dict[str, Any]] = []

    df = df.sort_values(["stock_id", "date"]).copy()
    for stock_id, group in df.groupby("stock_id"):
        group = group.reset_index(drop=True)
        x = group[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        y = group[label_column].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(group["date"])

        for idx in range(lookback - 1, len(group)):
            if np.isnan(y[idx]):
                continue
            seq = x[idx - lookback + 1 : idx + 1]
            sequences.append(seq)
            targets.append(float(y[idx]))
            metas.append({"stock_id": str(stock_id), "date": dates.iloc[idx], **_meta_context(group.iloc[idx])})

    if not sequences:
        return (
            np.empty((0, lookback, len(feature_columns)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.DataFrame(columns=["stock_id", "date"]),
        )

    return np.stack(sequences), np.array(targets, dtype=np.float32), pd.DataFrame(metas)


def train_lstm_regressor(
    dataset: SequenceDatasetBundle,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    batch_size: int = 512,
    epochs: int = 8,
    learning_rate: float = 1e-3,
    seed: int | None = None,
) -> tuple[Any, pd.DataFrame]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is required for the LSTM baseline. Install PyTorch before running the deep baseline."
        ) from exc

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LSTMRegressor(nn.Module):
        def __init__(self, input_size: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :]).squeeze(-1)

    model = LSTMRegressor(input_size=dataset.x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(dataset.x_train), torch.from_numpy(dataset.y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset.x_valid), batch_size):
            x_valid = torch.from_numpy(dataset.x_valid[start : start + batch_size]).to(device)
            preds.append(model(x_valid).detach().cpu().numpy())

    valid_pred_df = dataset.valid_meta.copy()
    valid_pred_df["score"] = np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)
    valid_pred_df["label"] = dataset.y_valid
    return model, valid_pred_df
