from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INDUSTRY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("bank", ("银行",)),
    ("broker", ("证券", "申万宏源", "中金公司", "东方财富")),
    ("insurance", ("保险",)),
    ("coal", ("煤",)),
    ("oil_gas", ("石油", "石化", "油服", "海油")),
    ("nonferrous", ("铜", "铝", "黄金", "稀土", "锂", "矿业", "资源", "钴", "钼", "钨")),
    ("steel", ("钢",)),
    ("power", ("电力", "电网", "水电", "火电", "核电", "能源")),
    ("utility", ("燃气", "环保", "水务")),
    ("transport", ("机场", "港", "航运", "高速", "铁路", "物流", "航空")),
    ("real_estate", ("地产", "置业")),
    ("construction", ("建筑", "建工", "交建", "铁建", "中铁")),
    ("materials", ("建材", "水泥", "玻纤")),
    ("chemical", ("化工", "化学", "新和成", "万华", "龙佰")),
    ("machinery", ("机械", "重工", "设备", "机器人")),
    ("auto", ("汽车", "汽", "客车", "轮胎", "赛力斯", "比亚迪", "潍柴")),
    ("appliance", ("电器", "家电", "美的", "格力", "海尔")),
    ("electronics", ("电子", "半导体", "微", "光电", "面板", "芯片", "中芯")),
    ("telecom", ("通信", "运营", "移动", "电信", "联通")),
    ("software_media", ("软件", "网络", "传媒", "游戏", "信息", "计算机")),
    ("healthcare", ("医药", "医疗", "生物", "制药", "疫苗", "医院")),
    ("consumer_staples", ("食品", "饮料", "乳业", "啤酒", "酿酒", "白酒", "伊利", "茅台", "五粮液", "泸州")),
    ("retail_consumer", ("商业", "零售", "百货", "免税", "旅游", "酒店")),
    ("agriculture", ("农业", "种业", "牧业", "养殖", "农牧")),
    ("military", ("军工", "船舶", "航发", "航空工业")),
]


def load_industry_map(stock_list_path: str | Path) -> pd.DataFrame:
    path = Path(stock_list_path)
    if not path.exists():
        return pd.DataFrame(columns=["stock_id", "industry_name", "industry_id", "code_name"])

    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})
    out = pd.DataFrame()
    out["stock_id"] = df["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("")
    out["code_name"] = df["code_name"].astype(str)
    out["industry_name"] = out["code_name"].map(infer_industry_from_name)
    industry_order = sorted(out["industry_name"].dropna().unique().tolist())
    industry_to_id = {name: idx for idx, name in enumerate(industry_order)}
    out["industry_id"] = out["industry_name"].map(industry_to_id).fillna(-1).astype(int)
    return out.drop_duplicates(subset=["stock_id"]).reset_index(drop=True)


def infer_industry_from_name(name: str) -> str:
    text = str(name)
    for industry, keywords in INDUSTRY_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return industry
    return "other"


def add_industry_features(
    df: pd.DataFrame,
    industry_map: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    if industry_map.empty:
        return df

    out = df.merge(
        industry_map[["stock_id", "industry_name", "industry_id", "code_name"]],
        on="stock_id",
        how="left",
    )
    out["industry_name"] = out["industry_name"].fillna("other")
    out["industry_id"] = out["industry_id"].fillna(-1).astype(int)

    industries = sorted(name for name in out["industry_name"].dropna().unique().tolist() if name != "other")
    onehot_cols = {
        f"industry_{industry}": (out["industry_name"] == industry).astype(float)
        for industry in industries
    }
    if onehot_cols:
        out = pd.concat([out, pd.DataFrame(onehot_cols, index=out.index)], axis=1)

    preferred_cols = [
        "ret_5",
        "ret_20",
        "stock_excess_ret_5",
        "stock_excess_ret_20",
        "ma_ratio_20",
        "volume_ratio_20",
        "beta_20",
        "idio_ret_20",
    ]
    base_cols = [col for col in preferred_cols if col in out.columns]

    grouped = out.groupby(["date", "industry_name"], observed=True)
    feature_blocks: dict[str, pd.Series] = {}
    for col in base_cols:
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std").replace(0, np.nan)
        rank = grouped[col].rank(pct=True)
        feature_blocks[f"industry_mean_{col}"] = mean
        feature_blocks[f"industry_excess_{col}"] = out[col] - mean
        feature_blocks[f"industry_z_{col}"] = ((out[col] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feature_blocks[f"industry_rank_{col}"] = rank.fillna(0.5)

    if feature_blocks:
        out = pd.concat([out, pd.DataFrame(feature_blocks, index=out.index)], axis=1)

    return out
