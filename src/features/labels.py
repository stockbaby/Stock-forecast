from __future__ import annotations

import pandas as pd


def add_forward_return_label(
    df: pd.DataFrame,
    label_name: str = "y_ret_5d_open_open",
    buy_offset: int = 1,
    sell_offset: int = 5,
    sell_fallback_offset: int | None = None,
) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("stock_id", group_keys=False)

    future_buy_open = g["open"].shift(-buy_offset)
    future_sell_open = g["open"].shift(-sell_offset)
    if sell_fallback_offset is not None:
        fallback_sell_open = g["open"].shift(-sell_fallback_offset)
        future_sell_open = future_sell_open.fillna(fallback_sell_open)
    out[label_name] = (future_sell_open / future_buy_open) - 1.0
    return out
