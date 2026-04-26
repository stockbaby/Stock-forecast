from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.deep_sequence import SequenceDatasetBundle


@dataclass
class ITransformerTrainConfig:
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    batch_size: int = 512
    epochs: int = 12
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 3
    lr_decay_factor: float = 0.5
    min_lr: float = 1e-5
    label_clip: float = 0.18
    eval_batch_size: int = 1024
    seed: int = 42


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        return


def _normalize_sequences(
    x_train: np.ndarray,
    x_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = x_train.copy()
    valid = x_valid.copy()
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train = np.clip((train - mean) / std, -6.0, 6.0)
    valid = np.clip((valid - mean) / std, -6.0, 6.0)
    return train.astype(np.float32), valid.astype(np.float32)


def _predict_in_batches(model, x_array: np.ndarray, batch_size: int, device, torch_module) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    with torch_module.no_grad():
        for start in range(0, len(x_array), batch_size):
            batch = torch_module.from_numpy(x_array[start : start + batch_size]).to(device)
            preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)


def train_itransformer_regressor(
    dataset: SequenceDatasetBundle,
    config: ITransformerTrainConfig,
) -> tuple[object, pd.DataFrame]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is required for the iTransformer backbone. Install PyTorch before running this script."
        ) from exc

    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, x_valid = _normalize_sequences(dataset.x_train, dataset.x_valid)
    y_train = np.clip(dataset.y_train.astype(np.float32), -config.label_clip, config.label_clip)
    y_valid = np.clip(dataset.y_valid.astype(np.float32), -config.label_clip, config.label_clip)

    seq_len = x_train.shape[1]
    num_features = x_train.shape[2]

    class ITransformerRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.series_projection = nn.Sequential(
                nn.LayerNorm(seq_len),
                nn.Linear(seq_len, config.d_model),
                nn.GELU(),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.feature_gate = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, config.dim_feedforward),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.dim_feedforward, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [batch, seq_len, features] -> variable tokens [batch, features, seq_len]
            tokens = x.transpose(1, 2)
            tokens = self.series_projection(tokens)
            encoded = self.encoder(tokens)
            gate = torch.softmax(self.feature_gate(encoded).squeeze(-1), dim=1)
            pooled = (encoded * gate.unsqueeze(-1)).sum(dim=1)
            feature_mean = encoded.mean(dim=1)
            fusion = torch.cat([pooled, feature_mean], dim=1)
            return self.head(fusion).squeeze(-1)

    model = ITransformerRegressor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.lr_decay_factor,
        patience=1,
        min_lr=config.min_lr,
    )
    criterion = nn.SmoothL1Loss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("-inf")
    epochs_without_improve = 0

    for _ in range(config.epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            reg_loss = criterion(preds, y_batch)
            pred_centered = preds - preds.mean()
            target_centered = y_batch - y_batch.mean()
            pred_std = pred_centered.std().clamp(min=1e-6)
            target_std = target_centered.std().clamp(min=1e-6)
            corr_loss = 1.0 - (pred_centered * target_centered).mean() / (pred_std * target_std)
            loss = 0.85 * reg_loss + 0.15 * corr_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        preds = _predict_in_batches(model, x_valid, config.eval_batch_size, device, torch)

        eval_df = dataset.valid_meta.copy()
        eval_df["score"] = preds
        eval_df["label"] = dataset.y_valid
        day_scores = []
        for _, group in eval_df.groupby("date"):
            if len(group) < 5:
                continue
            corr = group["score"].rank().corr(group["label"].rank(), method="spearman")
            if pd.notna(corr):
                day_scores.append(float(corr))
        val_score = float(np.mean(day_scores)) if day_scores else float("-inf")
        scheduler.step(val_score)

        if val_score > best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= config.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    preds = _predict_in_batches(model, x_valid, config.eval_batch_size, device, torch)

    valid_pred_df = dataset.valid_meta.copy()
    valid_pred_df["score"] = preds
    valid_pred_df["label"] = dataset.y_valid
    return model, valid_pred_df
