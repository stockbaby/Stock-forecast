from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


@dataclass
class FittedModel:
    name: str
    model: Any
    feature_columns: list[str]


def fit_baseline_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model_type: str = "lightgbm",
) -> FittedModel:
    x_train = train_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_df[label_column].astype(float)

    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
            model.fit(x_train, y_train)
            return FittedModel(name="lightgbm", model=model, feature_columns=feature_columns)
        except ImportError:
            pass

    if model_type == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=1024,
            learning_rate_init=1e-3,
            max_iter=50,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        model.fit(x_train, y_train)
        return FittedModel(name="mlp", model=model, feature_columns=feature_columns)

    fallback = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=42,
    )
    fallback.fit(x_train, y_train)
    return FittedModel(name="hist_gbdt", model=fallback, feature_columns=feature_columns)


def predict_scores(model: FittedModel, df: pd.DataFrame) -> pd.Series:
    x = df[model.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.Series(model.model.predict(x), index=df.index, name="score")
