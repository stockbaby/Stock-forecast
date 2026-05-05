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
class MasterTrainConfig:
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    batch_size: int = 384
    epochs: int = 12
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 4
    lr_decay_factor: float = 0.5
    min_lr: float = 1e-5
    market_gate_strength: float = 1.0
    regression_weight: float = 0.7
    rank_weight: float = 0.2
    corr_weight: float = 0.1
    official_rank_weight: float = 0.0
    official_top_k: int = 5
    official_top_k_weight: float = 2.0
    official_base_weight: float = 1.0
    official_temperature: float = 1.0
    date_batching: bool = False
    validation_strategy: str = "proportional_positive_thr0.0"
    validation_rank_weight: float = 0.1
    label_clip: float = 0.18
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


def _normalize_by_group(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    group_indices: list[tuple[str, list[int]]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[list[int], np.ndarray, np.ndarray]]]:
    stats = _fit_group_normalizers(x_train, group_indices)
    return _apply_group_normalizers(x_train, stats), _apply_group_normalizers(x_valid, stats), stats


def _fit_group_normalizers(
    x_train: np.ndarray,
    group_indices: list[tuple[str, list[int]]],
) -> list[tuple[list[int], np.ndarray, np.ndarray]]:
    stats: list[tuple[list[int], np.ndarray, np.ndarray]] = []
    for _, indices in group_indices:
        train_slice = x_train[:, :, indices]
        mean = train_slice.mean(axis=(0, 1), keepdims=True)
        std = train_slice.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        stats.append((indices, mean.astype(np.float32), std.astype(np.float32)))
    return stats


def _apply_group_normalizers(
    x_array: np.ndarray,
    stats: list[tuple[list[int], np.ndarray, np.ndarray]],
) -> np.ndarray:
    normalized = x_array.copy()
    for indices, mean, std in stats:
        normalized[:, :, indices] = np.clip((normalized[:, :, indices] - mean) / std, -6.0, 6.0)
    return normalized.astype(np.float32)


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


def _build_topk_sample_weights(
    target,
    top_k: int,
    top_k_weight: float,
    base_weight: float,
    torch_module,
):
    weights = torch_module.full_like(target, float(base_weight))
    if target.numel() == 0:
        return weights
    k = min(int(top_k), int(target.numel()))
    if k <= 0:
        return weights
    _, top_indices = torch_module.topk(target, k)
    weights[top_indices] = float(top_k_weight)
    return weights


def _weighted_listwise_loss(pred, target, weights, temperature: float, torch_module) -> object:
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    temp = max(float(temperature), 1e-3)
    pred_probs = torch_module.softmax(pred / temp, dim=0)
    target_probs = torch_module.softmax(target / temp, dim=0)
    weighted_ce = -(target_probs * torch_module.log(pred_probs + 1e-12) * weights)
    return weighted_ce.sum() / weights.sum().clamp(min=1e-6)


def _weighted_pairwise_rank_loss(pred, target, weights, torch_module) -> object:
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    sign = torch_module.sign(target_diff)
    mask = sign != 0
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    pair_weights = (weights.unsqueeze(1) + weights.unsqueeze(0)) * 0.5
    margin = torch_module.abs(target_diff).clamp(min=1e-3, max=0.1)
    loss = torch_module.nn.functional.softplus(-(pred_diff * sign)) * margin * pair_weights
    return loss[mask].mean()


def _official_weighted_rank_loss(pred, target, config: MasterTrainConfig, torch_module) -> object:
    weights = _build_topk_sample_weights(
        target=target,
        top_k=config.official_top_k,
        top_k_weight=config.official_top_k_weight,
        base_weight=config.official_base_weight,
        torch_module=torch_module,
    )
    listwise = _weighted_listwise_loss(
        pred=pred,
        target=target,
        weights=weights,
        temperature=config.official_temperature,
        torch_module=torch_module,
    )
    pairwise = _weighted_pairwise_rank_loss(
        pred=pred,
        target=target,
        weights=weights,
        torch_module=torch_module,
    )
    return listwise + pairwise


def _daily_group_rank_loss(pred, target, date_codes, config: MasterTrainConfig, torch_module) -> object:
    losses = []
    for code in torch_module.unique(date_codes):
        mask = date_codes == code
        if int(mask.sum().item()) < 2:
            continue
        losses.append(_official_weighted_rank_loss(pred[mask], target[mask], config, torch_module))
    if not losses:
        return pred.new_tensor(0.0)
    return torch_module.stack(losses).mean()


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


def train_master_regressor(
    dataset: SequenceDatasetBundle,
    config: MasterTrainConfig,
) -> tuple[object, pd.DataFrame]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for MASTER-style training.") from exc

    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_columns = dataset.feature_columns
    groups = build_feature_groups(feature_columns)
    group_indices = [
        (name, [feature_columns.index(col) for col in cols])
        for name, cols in groups.items()
        if cols
    ]
    x_train, x_valid, normalizer_stats = _normalize_by_group(dataset.x_train, dataset.x_valid, group_indices)
    y_train = np.clip(dataset.y_train.astype(np.float32), -config.label_clip, config.label_clip)
    y_valid = np.clip(dataset.y_valid.astype(np.float32), -config.label_clip, config.label_clip)

    market_cols = [
        idx
        for idx, col in enumerate(feature_columns)
        if any(token in col for token in ("index_", "market_", "stock_excess", "beta_", "regime_", "idio_", "style_"))
    ]
    stock_cols = [idx for idx in range(len(feature_columns)) if idx not in market_cols]
    if not market_cols:
        market_cols = stock_cols[: min(8, len(stock_cols))]

    class MasterRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stock_proj = nn.Sequential(
                nn.LayerNorm(len(stock_cols)),
                nn.Linear(len(stock_cols), config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.market_proj = nn.Sequential(
                nn.LayerNorm(len(market_cols)),
                nn.Linear(len(market_cols), config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.stock_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.market_encoder = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.market_gate = nn.Sequential(
                nn.LayerNorm(config.hidden_dim * 2),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Sigmoid(),
            )
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.time_attention = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim * 2),
                nn.Linear(config.hidden_dim * 2, config.ff_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.ff_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            stock_x = self.stock_proj(x[:, :, stock_cols])
            market_x = self.market_proj(x[:, :, market_cols])
            stock_repr = self.stock_encoder(stock_x)
            market_repr, market_hidden = self.market_encoder(market_x)
            market_state = market_hidden[-1]

            repeated_market = market_state.unsqueeze(1).expand(-1, stock_repr.size(1), -1)
            gate = self.market_gate(torch.cat([stock_repr, repeated_market], dim=-1))
            guided = stock_repr * (1.0 + config.market_gate_strength * gate)

            attn_out, _ = self.cross_attention(guided, guided, guided)
            fused = guided + attn_out
            time_weights = torch.softmax(self.time_attention(fused).squeeze(-1), dim=1)
            pooled = (fused * time_weights.unsqueeze(-1)).sum(dim=1)
            combined = torch.cat([pooled, market_state], dim=-1)
            return self.head(combined).squeeze(-1)

    model = MasterRegressor().to(device)
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

    train_dates = pd.to_datetime(dataset.train_meta["date"]).astype("int64").to_numpy()
    _, train_date_codes = np.unique(train_dates, return_inverse=True)
    train_tensor_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(train_date_codes.astype(np.int64)),
    )
    if config.date_batching:
        rng = np.random.default_rng(config.seed)
        batches = [
            np.asarray(indices, dtype=np.int64)
            for indices in pd.Series(np.arange(len(train_date_codes))).groupby(train_date_codes).apply(list).tolist()
        ]
        rng.shuffle(batches)
        train_loader = DataLoader(train_tensor_dataset, batch_sampler=batches)
    else:
        train_loader = DataLoader(
            train_tensor_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("-inf")
    patience = 0

    for _ in range(config.epochs):
        model.train()
        for x_batch, y_batch, date_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            date_batch = date_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            reg_loss = torch.nn.functional.smooth_l1_loss(preds, y_batch)
            rank_loss = _pairwise_rank_loss(preds, y_batch, torch)
            corr_loss = _corr_loss(preds, y_batch, torch)
            if config.date_batching:
                official_rank_loss = _daily_group_rank_loss(preds, y_batch, date_batch, config, torch)
            else:
                official_rank_loss = _official_weighted_rank_loss(preds, y_batch, config, torch)
            loss = (
                config.regression_weight * reg_loss
                + config.rank_weight * rank_loss
                + config.corr_weight * corr_loss
                + config.official_rank_weight * official_rank_loss
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
            strategy=config.validation_strategy,
            top_k=5,
            max_weight_sum=1.0,
            temperature=0.8,
        )
        val_score = float(portfolio_eval["mean_return"]) + config.validation_rank_weight * float(val_rank_ic)
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
    preds = _predict_in_batches(model, x_valid, config.batch_size, device, torch)
    valid_pred_df = dataset.valid_meta.copy()
    valid_pred_df["score"] = preds
    valid_pred_df["label"] = dataset.y_valid
    x_infer = getattr(dataset, "x_infer", None)
    infer_meta = getattr(dataset, "infer_meta", None)
    if x_infer is not None and infer_meta is not None and len(x_infer) > 0:
        x_infer = _apply_group_normalizers(x_infer, normalizer_stats)
        infer_preds = _predict_in_batches(model, x_infer, config.batch_size, device, torch)
        infer_pred_df = infer_meta.copy()
        infer_pred_df["score"] = infer_preds
        model.infer_pred_df = infer_pred_df
    return model, valid_pred_df
