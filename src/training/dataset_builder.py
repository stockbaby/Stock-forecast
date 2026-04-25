from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.io import load_price_data, save_dataframe
from src.features.alpha_factors import add_basic_price_features, add_cross_sectional_features
from src.features.labels import add_forward_return_label


@dataclass
class DatasetBuildConfig:
    raw_dir: str
    processed_path: str
    windows: list[int]
    label_name: str
    buy_offset: int
    sell_offset: int
    sell_fallback_offset: int | None = None


def build_model_dataset(config: DatasetBuildConfig) -> pd.DataFrame:
    df = load_price_data(config.raw_dir)
    df = add_basic_price_features(df, config.windows)
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
        *[f"ret_{window}" for window in config.windows],
        *[f"volatility_{window}" for window in config.windows],
        *[f"ma_ratio_{window}" for window in config.windows],
        *[f"volume_ratio_{window}" for window in config.windows],
    ]
    df = add_cross_sectional_features(df, base_feature_cols)
    save_dataframe(df, config.processed_path)
    return df
