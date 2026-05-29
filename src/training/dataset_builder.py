from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.io import load_price_data, save_dataframe
from src.features.alpha_factors import add_basic_price_features, add_cross_sectional_features
from src.features.industry_context import add_industry_features, load_industry_map
from src.features.labels import add_forward_return_label
from src.features.market_context import add_market_index_features, load_market_index_frame


@dataclass
class DatasetBuildConfig:
    raw_dir: str
    processed_path: str
    market_index_path: str | None
    industry_map_path: str | None
    windows: list[int]
    label_name: str
    buy_offset: int
    sell_offset: int
    sell_fallback_offset: int | None = None


def build_model_dataset(config: DatasetBuildConfig) -> pd.DataFrame:
    df = load_price_data(config.raw_dir)
    df = add_basic_price_features(df, config.windows)
    if config.market_index_path and Path(config.market_index_path).exists():
        index_df = load_market_index_frame(config.market_index_path)
        df = add_market_index_features(df, index_df, config.windows)
    if config.industry_map_path and Path(config.industry_map_path).exists():
        industry_df = load_industry_map(config.industry_map_path)
        df = add_industry_features(df, industry_df, config.windows)
    df = add_forward_return_label(
        df,
        label_name=config.label_name,
        buy_offset=config.buy_offset,
        sell_offset=config.sell_offset,
        sell_fallback_offset=config.sell_fallback_offset,
    )

    base_feature_cols = [
        "ret_1",
        "open_to_close",
        "high_to_low",
        "amplitude",
        "gap_open",
        "close_vs_prev_close",
        "body_ratio",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "close_location",
        "open_location",
        "amount_per_volume",
        "turnover_rate",
        "pct_chg_decimal",
        "amplitude_pct_decimal",
        "ema_12_ratio",
        "ema_26_ratio",
        "macd_line",
        "signal_line",
        "macd_hist",
        "rsi_14",
        *[f"ret_{window}" for window in config.windows],
        *[f"volatility_{window}" for window in config.windows],
        *[f"ret_mean_{window}" for window in config.windows],
        *[f"ret_std_{window}" for window in config.windows],
        *[f"ret_skew_{window}" for window in config.windows],
        *[f"ma_ratio_{window}" for window in config.windows],
        *[f"volume_ratio_{window}" for window in config.windows],
        *[f"high_ratio_{window}" for window in config.windows],
        *[f"close_to_high_{window}" for window in config.windows],
        *[f"breakout_strength_{window}" for window in config.windows],
        *[f"low_ratio_{window}" for window in config.windows],
        *[f"price_position_{window}" for window in config.windows],
        *[f"ret_to_vol_{window}" for window in config.windows],
        *[f"amount_ratio_{window}" for window in config.windows],
        *[f"turnover_ratio_{window}" for window in config.windows],
        *[f"pct_chg_mean_{window}" for window in config.windows],
        *[f"volume_volatility_{window}" for window in config.windows],
        *[f"volume_breakout_{window}" for window in config.windows],
        *[f"momentum_accel_{window}" for window in config.windows],
        "short_momentum_3_5",
        "short_momentum_accel_3_5",
        "short_volume_momentum_3_5",
        "amount_accel_3_5",
        "turnover_accel_3_5",
        "trend_alignment_5_20",
        "trend_accel_5_20",
        "breakout_volume_confirm_20",
        "limit_up_proximity",
        "fragile_rally_penalty",
    ]
    market_feature_cols = [
        "index_ret_1",
        "stock_excess_ret_1",
        "index_open_to_close",
        "index_high_to_low",
        "regime_trend",
        "regime_vol_ratio",
        "regime_drawdown",
        "regime_score",
        "regime_is_trending",
        "regime_is_high_vol",
        "beta_20",
        "beta_60",
        "idio_ret_20",
        "idio_ret_60",
        "style_liquidity_bucket",
        "style_vol_bucket",
        "style_beta_bucket",
        *[f"index_ret_{window}" for window in config.windows],
        *[f"index_ma_ratio_{window}" for window in config.windows],
        *[f"index_volatility_{window}" for window in config.windows],
        *[f"index_drawdown_{window}" for window in config.windows],
        *[f"stock_excess_ret_{window}" for window in config.windows],
        "short_excess_strength_3_5_10",
        *[f"style_excess_ret_{window}" for window in config.windows],
        *[f"style_z_ret_{window}" for window in config.windows],
        *[f"style_rank_ret_{window}" for window in config.windows],
        *[f"style_excess_stock_excess_ret_{window}" for window in config.windows],
        *[f"style_z_stock_excess_ret_{window}" for window in config.windows],
        *[f"style_rank_stock_excess_ret_{window}" for window in config.windows],
        "style_excess_ma_ratio_5",
        "style_z_ma_ratio_5",
        "style_rank_ma_ratio_5",
        "style_excess_ma_ratio_20",
        "style_z_ma_ratio_20",
        "style_rank_ma_ratio_20",
        "style_excess_beta_20",
        "style_z_beta_20",
        "style_rank_beta_20",
        "style_excess_idio_ret_20",
        "style_z_idio_ret_20",
        "style_rank_idio_ret_20",
    ]
    industry_feature_cols = [
        col
        for col in ["industry_id", *[name for name in df.columns if name.startswith("industry_")]]
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    feature_candidates = [col for col in base_feature_cols if col in df.columns] + [
        col for col in market_feature_cols if col in df.columns
    ] + [col for col in industry_feature_cols if col in df.columns]
    df = add_cross_sectional_features(df, feature_candidates)
    save_dataframe(df, config.processed_path)
    return df
