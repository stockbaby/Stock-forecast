from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.alpha_factors import build_feature_groups
from src.models.deep_sequence import SequenceDatasetBundle
from src.portfolio.construct import evaluate_portfolio_strategy
from src.training.metrics import rank_ic


@dataclass
class StockMixerTrainConfig:
    hidden_dim: int = 128
    mixer_dim: int = 256
    temporal_dim: int = 64
    dropout: float = 0.15
    batch_size: int = 384
    epochs: int = 16
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 4
    lr_decay_factor: float = 0.5
    min_lr: float = 1e-5
    recent_weight_power: float = 1.8
    regression_weight: float = 0.7
    rank_weight: float = 0.2
    corr_weight: float = 0.1
    label_clip: float = 0.2
    patch_sizes: tuple[int, ...] = (5, 10, 20, 30)
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


def _group_feature_indices(feature_columns: list[str]) -> list[tuple[str, list[int]]]:
    groups = build_feature_groups(feature_columns)
    return [
        (name, [feature_columns.index(col) for col in cols])
        for name, cols in groups.items()
        if cols
    ]


def _normalize_by_group(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    group_indices: list[tuple[str, list[int]]],
) -> tuple[np.ndarray, np.ndarray]:
    train = x_train.copy()
    valid = x_valid.copy()

    for _, indices in group_indices:
        train_slice = train[:, :, indices]
        mean = train_slice.mean(axis=(0, 1), keepdims=True)
        std = train_slice.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)

        train[:, :, indices] = np.clip((train_slice - mean) / std, -6.0, 6.0)
        valid[:, :, indices] = np.clip((valid[:, :, indices] - mean) / std, -6.0, 6.0)

    return train.astype(np.float32), valid.astype(np.float32)


def _date_batches(meta: pd.DataFrame, batch_size: int, shuffle: bool) -> list[np.ndarray]:
    grouped = meta.reset_index(drop=True).groupby("date").indices
    dates = list(grouped.keys())
    if shuffle:
        random.shuffle(dates)

    batches: list[np.ndarray] = []
    for date in dates:
        indices = np.asarray(grouped[date], dtype=np.int64)
        if len(indices) <= batch_size:
            batches.append(indices)
            continue
        for start in range(0, len(indices), batch_size):
            batches.append(indices[start : start + batch_size])
    return batches


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
    corr = (pred_centered * target_centered).mean() / (pred_std * target_std)
    return 1.0 - corr


def _predict_in_batches(model, x_array: np.ndarray, batch_size: int, device, torch_module) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    with torch_module.no_grad():
        for start in range(0, len(x_array), batch_size):
            batch = torch_module.from_numpy(x_array[start : start + batch_size]).to(device)
            preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)


def train_stockmixer_regressor(
    dataset: SequenceDatasetBundle,
    config: StockMixerTrainConfig,
) -> tuple[object, pd.DataFrame]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is required for the StockMixer baseline. Install PyTorch before running this script."
        ) from exc

    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_columns = dataset.feature_columns
    group_indices = _group_feature_indices(feature_columns)
    x_train, x_valid = _normalize_by_group(dataset.x_train, dataset.x_valid, group_indices)
    y_train = np.clip(dataset.y_train.astype(np.float32), -config.label_clip, config.label_clip)
    y_valid = np.clip(dataset.y_valid.astype(np.float32), -config.label_clip, config.label_clip)
    seq_len = x_train.shape[1]
    patch_sizes = tuple(sorted({min(p, seq_len) for p in config.patch_sizes if p > 0}))

    class GroupFeatureProjector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.group_names = [name for name, _ in group_indices]
            self.group_indices = [idx for _, idx in group_indices]
            self.projectors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(len(indices)),
                        nn.Linear(len(indices), config.hidden_dim),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                    )
                    for _, indices in group_indices
                ]
            )
            self.fuse = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            projected = []
            for indices, projector in zip(self.group_indices, self.projectors):
                projected.append(projector(x[:, :, indices]))
            stacked = torch.stack(projected, dim=0).mean(dim=0)
            return self.fuse(stacked)

    class MixerLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.channel_norm = nn.LayerNorm(config.hidden_dim)
            self.channel_mlp = nn.Sequential(
                nn.Linear(config.hidden_dim, config.mixer_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.mixer_dim, config.hidden_dim),
            )
            self.token_norm = nn.LayerNorm(config.hidden_dim)
            self.token_mlp = nn.Sequential(
                nn.Linear(seq_len, config.temporal_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.temporal_dim, seq_len),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.channel_mlp(self.channel_norm(x))
            token_view = self.token_norm(x).transpose(1, 2)
            token_view = self.token_mlp(token_view).transpose(1, 2)
            return x + token_view

    class StockMixerRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_projector = GroupFeatureProjector()
            self.mixer_layers = nn.ModuleList([MixerLayer(), MixerLayer(), MixerLayer()])
            self.patch_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(config.hidden_dim),
                        nn.Linear(config.hidden_dim, config.hidden_dim),
                        nn.GELU(),
                    )
                    for _ in patch_sizes
                ]
            )
            fusion_dim = config.hidden_dim * (2 + len(patch_sizes))
            self.head = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, config.mixer_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.mixer_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

        def _recent_pool(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
            patch = x[:, -patch_size:, :]
            weights = torch.linspace(1.0 / patch_size, 1.0, patch_size, device=x.device)
            weights = weights.pow(config.recent_weight_power)
            weights = weights / weights.sum()
            return (patch * weights.view(1, patch_size, 1)).sum(dim=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.feature_projector(x)
            for layer in self.mixer_layers:
                x = layer(x)

            global_weights = torch.linspace(1.0 / seq_len, 1.0, seq_len, device=x.device)
            global_weights = global_weights.pow(config.recent_weight_power)
            global_weights = global_weights / global_weights.sum()
            global_pool = (x * global_weights.view(1, seq_len, 1)).sum(dim=1)
            last_token = x[:, -1, :]

            patch_vectors = [
                projection(self._recent_pool(x, patch_size))
                for patch_size, projection in zip(patch_sizes, self.patch_projections)
            ]
            fusion = torch.cat([last_token, global_pool, *patch_vectors], dim=1)
            return self.head(fusion).squeeze(-1)

    model = StockMixerRegressor().to(device)
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
    reg_criterion = nn.SmoothL1Loss()

    train_batches = _date_batches(dataset.train_meta, config.batch_size, shuffle=True)
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("-inf")
    epochs_without_improve = 0

    for _ in range(config.epochs):
        model.train()
        shuffled_batches = train_batches.copy()
        random.shuffle(shuffled_batches)
        for batch_indices in shuffled_batches:
            x_batch = torch.from_numpy(x_train[batch_indices]).to(device)
            y_batch = torch.from_numpy(y_train[batch_indices]).to(device)
            optimizer.zero_grad()
            preds = model(x_batch)

            reg_loss = reg_criterion(preds, y_batch)
            rank_loss = _pairwise_rank_loss(preds, y_batch, torch)
            corr_loss = _corr_loss(preds, y_batch, torch)
            loss = (
                config.regression_weight * reg_loss
                + config.rank_weight * rank_loss
                + config.corr_weight * corr_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        preds = _predict_in_batches(model, x_valid, config.batch_size, device, torch)
        eval_df = dataset.valid_meta.copy()
        eval_df["score"] = preds
        eval_df["label"] = y_valid
        val_rank_ic = rank_ic(eval_df, "label", "score")
        portfolio_eval = evaluate_portfolio_strategy(
            eval_df,
            label_col="label",
            score_col="score",
            strategy="proportional_positive_thr0.0",
            top_k=5,
            max_weight_sum=1.0,
            temperature=0.8,
        )
        val_score = float(portfolio_eval["mean_return"]) + 0.1 * float(val_rank_ic)
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
    preds = _predict_in_batches(model, x_valid, config.batch_size, device, torch)
    valid_pred_df = dataset.valid_meta.copy()
    valid_pred_df["score"] = preds
    valid_pred_df["label"] = dataset.y_valid
    return model, valid_pred_df
