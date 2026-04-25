from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.io import load_price_data, save_dataframe
from src.features.alpha_factors import add_basic_price_features, add_cross_sectional_features
from src.features.labels import add_forward_return_label
from src.features.market_context import add_market_index_features, load_market_index_frame


@dataclass
class DatasetBuildConfig:
    raw_dir: str
    processed_path: str
    market_index_path: str | None
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
        *[f"low_ratio_{window}" for window in config.windows],
        *[f"price_position_{window}" for window in config.windows],
        *[f"ret_to_vol_{window}" for window in config.windows],
        *[f"amount_ratio_{window}" for window in config.windows],
        *[f"turnover_ratio_{window}" for window in config.windows],
        *[f"pct_chg_mean_{window}" for window in config.windows],
        *[f"volume_volatility_{window}" for window in config.windows],
    ]
    market_feature_cols = [
        "index_ret_1",
        "stock_excess_ret_1",
        "index_open_to_close",
        "index_high_to_low",
        *[f"index_ret_{window}" for window in config.windows],
        *[f"index_ma_ratio_{window}" for window in config.windows],
        *[f"stock_excess_ret_{window}" for window in config.windows],
    ]
    feature_candidates = [col for col in base_feature_cols if col in df.columns] + [
        col for col in market_feature_cols if col in df.columns
    ]
    df = add_cross_sectional_features(df, feature_candidates)
    save_dataframe(df, config.processed_path)
    return df
