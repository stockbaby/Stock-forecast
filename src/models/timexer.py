from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.deep_sequence import SequenceDatasetBundle
from src.portfolio.construct import evaluate_portfolio_strategy
from src.training.metrics import rank_ic


@dataclass
class TimeXerTrainConfig:
    d_model: int = 96
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 192
    patch_len: int = 5
    stride: int = 5
    dropout: float = 0.1
    batch_size: int = 512
    epochs: int = 6
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 2
    lr_decay_factor: float = 0.5
    min_lr: float = 1e-5
    regression_weight: float = 0.65
    rank_weight: float = 0.2
    corr_weight: float = 0.15
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


def _normalize_sequences(x_train: np.ndarray, x_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train = np.clip((x_train - mean) / std, -6.0, 6.0)
    valid = np.clip((x_valid - mean) / std, -6.0, 6.0)
    return train.astype(np.float32), valid.astype(np.float32)


def _pairwise_rank_loss(pred, target, torch_module) -> object:
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    sign = torch_module.sign(target_diff)
    mask = sign != 0
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    margin = torch_module.abs(target_diff).clamp(min=1e-3, max=0.1)
    loss = torch_module.nn.functional.softplus(-(pred_diff * sign)) * margin
    return loss[mask].mean()


def _corr_loss(pred, target, torch_module) -> object:
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    pred_std = pred_centered.std().clamp(min=1e-6)
    target_std = target_centered.std().clamp(min=1e-6)
    return 1.0 - (pred_centered * target_centered).mean() / (pred_std * target_std)


def _date_batches(meta: pd.DataFrame, batch_size: int, shuffle: bool) -> list[np.ndarray]:
    rng = np.random.default_rng()
    batches: list[np.ndarray] = []
    for _, group in meta.reset_index().groupby("date"):
        indices = group["index"].to_numpy().copy()
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batches.append(indices[start : start + batch_size])
    if shuffle:
        rng.shuffle(batches)
    return batches


def _predict_in_batches(model, x_array: np.ndarray, batch_size: int, device, torch_module) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    with torch_module.no_grad():
        for start in range(0, len(x_array), batch_size):
            batch = torch_module.from_numpy(x_array[start : start + batch_size]).to(device)
            preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)


def train_timexer_regressor(
    dataset: SequenceDatasetBundle,
    config: TimeXerTrainConfig,
) -> tuple[object, pd.DataFrame]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for TimeXer-style training.") from exc

    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, x_valid = _normalize_sequences(dataset.x_train, dataset.x_valid)
    y_train = np.clip(dataset.y_train.astype(np.float32), -config.label_clip, config.label_clip)

    feature_columns = dataset.feature_columns
    exog_indices = [
        idx
        for idx, col in enumerate(feature_columns)
        if any(token in col for token in ("index_", "regime_", "stock_excess", "beta_", "idio_", "style_"))
    ]
    endog_indices = [idx for idx in range(len(feature_columns)) if idx not in exog_indices]
    if not exog_indices:
        exog_indices = endog_indices[: min(8, len(endog_indices))]
    if not endog_indices:
        endog_indices = exog_indices

    seq_len = x_train.shape[1]
    patch_len = min(config.patch_len, seq_len)
    stride = max(1, config.stride)
    patch_starts = list(range(0, max(seq_len - patch_len + 1, 1), stride))
    if patch_starts[-1] != seq_len - patch_len:
        patch_starts.append(seq_len - patch_len)
    num_patches = len(patch_starts)

    class TimeXerRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_projection = nn.Sequential(
                nn.LayerNorm(patch_len * len(endog_indices)),
                nn.Linear(patch_len * len(endog_indices), config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.exog_projection = nn.Sequential(
                nn.LayerNorm(len(exog_indices) * 3),
                nn.Linear(len(exog_indices) * 3, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.position = nn.Parameter(torch.zeros(1, num_patches, config.d_model))
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.nhead,
                dropout=config.dropout,
                batch_first=True,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, config.dim_feedforward),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.dim_feedforward, 1),
            )

        def forward(self, x):
            endog = x[:, :, endog_indices]
            patches = []
            for start in patch_starts:
                patch = endog[:, start : start + patch_len, :].reshape(endog.shape[0], -1)
                patches.append(self.patch_projection(patch))
            tokens = torch.stack(patches, dim=1) + self.position

            exog = x[:, :, exog_indices]
            exog_state = torch.cat([exog[:, -1, :], exog.mean(dim=1), exog.std(dim=1)], dim=1)
            exog_token = self.exog_projection(exog_state).unsqueeze(1)
            crossed, _ = self.cross_attn(tokens, exog_token, exog_token)
            encoded = self.encoder(tokens + crossed)
            pooled = encoded.mean(dim=1)
            last_patch = encoded[:, -1, :]
            return self.head(torch.cat([pooled, last_patch], dim=1)).squeeze(-1)

    model = TimeXerRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.lr_decay_factor,
        patience=1,
        min_lr=config.min_lr,
    )
    criterion = nn.SmoothL1Loss()
    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)
    train_batches = _date_batches(dataset.train_meta, config.batch_size, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("-inf")
    patience = 0
    for _ in range(config.epochs):
        model.train()
        for indices in train_batches:
            x_batch = x_train_tensor[indices].to(device)
            y_batch = y_train_tensor[indices].to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = (
                config.regression_weight * criterion(preds, y_batch)
                + config.rank_weight * _pairwise_rank_loss(preds, y_batch, torch)
                + config.corr_weight * _corr_loss(preds, y_batch, torch)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        preds = _predict_in_batches(model, x_valid, config.eval_batch_size, device, torch)
        eval_df = dataset.valid_meta.copy()
        eval_df["score"] = preds
        eval_df["label"] = dataset.y_valid
        portfolio_eval = evaluate_portfolio_strategy(
            eval_df,
            label_col="label",
            score_col="score",
            strategy="softmax_t0.6",
            top_k=5,
            max_weight_sum=1.0,
            temperature=0.8,
        )
        val_score = float(portfolio_eval["mean_return"]) + 0.1 * float(rank_ic(eval_df, "label", "score"))
        scheduler.step(val_score)
        if val_score > best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    preds = _predict_in_batches(model, x_valid, config.eval_batch_size, device, torch)
    valid_pred_df = dataset.valid_meta.copy()
    valid_pred_df["score"] = preds
    valid_pred_df["label"] = dataset.y_valid
    return model, valid_pred_df
