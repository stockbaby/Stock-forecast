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

STOCK_NAME_OVERRIDES: dict[str, str] = {
    "华能国际": "power",
    "华电国际": "power",
    "国电南瑞": "power",
    "东方电气": "power",
    "华电新能": "power",
    "隆基绿能": "power",
    "中国广核": "power",
    "通威股份": "power",
    "中远海能": "transport",
    "中国东航": "transport",
    "圆通速递": "transport",
    "中国国航": "transport",
    "中国中车": "transport",
    "京沪高铁": "transport",
    "招商轮船": "transport",
    "中远海控": "transport",
    "国货航": "transport",
    "招商公路": "transport",
    "四川路桥": "construction",
    "中国中冶": "construction",
    "中国电建": "construction",
    "中国能建": "construction",
    "保利发展": "real_estate",
    "万科A": "real_estate",
    "招商蛇口": "real_estate",
    "国投资本": "broker",
    "中信建投": "broker",
    "国泰海通": "broker",
    "国联民生": "broker",
    "中国银河": "broker",
    "同花顺": "broker",
    "指南针": "broker",
    "中国平安": "insurance",
    "中国人保": "insurance",
    "中国太保": "insurance",
    "中国人寿": "insurance",
    "渝农商行": "bank",
    "沪农商行": "bank",
    "中国神华": "coal",
    "新奥股份": "utility",
    "特变电工": "power",
    "思源电气": "power",
    "国轩高科": "auto",
    "宁德时代": "auto",
    "德赛西威": "auto",
    "拓普集团": "auto",
    "福耀玻璃": "auto",
    "三花智控": "appliance",
    "公牛集团": "appliance",
    "石头科技": "appliance",
    "安克创新": "appliance",
    "同仁堂": "healthcare",
    "片仔癀": "healthcare",
    "药明康德": "healthcare",
    "百利天恒": "healthcare",
    "云南白药": "healthcare",
    "长春高新": "healthcare",
    "华润三九": "healthcare",
    "上海莱士": "healthcare",
    "科伦药业": "healthcare",
    "爱尔眼科": "healthcare",
    "康龙化成": "healthcare",
    "新产业": "healthcare",
    "爱美客": "healthcare",
    "山西汾酒": "consumer_staples",
    "海天味业": "consumer_staples",
    "今世缘": "consumer_staples",
    "古井贡酒": "consumer_staples",
    "新希望": "consumer_staples",
    "双汇发展": "consumer_staples",
    "洋河股份": "consumer_staples",
    "海大集团": "consumer_staples",
    "牧原股份": "consumer_staples",
    "温氏股份": "consumer_staples",
    "金龙鱼": "consumer_staples",
    "华利集团": "retail_consumer",
    "小商品城": "retail_consumer",
    "中国中免": "retail_consumer",
    "巨化股份": "chemical",
    "华鲁恒升": "chemical",
    "东方盛虹": "chemical",
    "盐湖股份": "chemical",
    "天赐材料": "chemical",
    "合盛硅业": "materials",
    "中国巨石": "materials",
    "山金国际": "nonferrous",
    "中油资本": "oil_gas",
    "恒立液压": "machinery",
    "中联重科": "machinery",
    "晶盛机电": "machinery",
    "中航机载": "military",
    "中国动力": "military",
    "中航沈飞": "military",
    "中航西飞": "military",
    "光启技术": "military",
    "中航成飞": "military",
    "中天科技": "telecom",
    "中国卫通": "telecom",
    "中国通号": "telecom",
    "中兴通讯": "telecom",
    "中际旭创": "telecom",
    "新易盛": "telecom",
    "生益科技": "electronics",
    "长电科技": "electronics",
    "工业富联": "electronics",
    "TCL科技": "electronics",
    "京东方A": "electronics",
    "沪电股份": "electronics",
    "北方华创": "electronics",
    "东山精密": "electronics",
    "歌尔股份": "electronics",
    "晶澳科技": "electronics",
    "立讯精密": "electronics",
    "领益智造": "electronics",
    "深南电路": "electronics",
    "鹏鼎控股": "electronics",
    "三环集团": "electronics",
    "蓝思科技": "electronics",
    "胜宏科技": "electronics",
    "圣邦股份": "electronics",
    "华勤技术": "electronics",
    "豪威集团": "electronics",
    "兆易创新": "electronics",
    "澜起科技": "electronics",
    "龙芯中科": "electronics",
    "盛美上海": "electronics",
    "沪硅产业": "electronics",
    "寒武纪": "electronics",
    "阿特斯": "electronics",
    "科大讯飞": "software_media",
    "大华股份": "software_media",
    "海康威视": "software_media",
    "紫光股份": "software_media",
    "三六零": "software_media",
    "中科曙光": "software_media",
    "金山办公": "software_media",
    "软通动力": "software_media",
    "华大九天": "software_media",
    "芒果超媒": "software_media",
    "昆仑万维": "software_media",
    "润泽科技": "software_media",
    "传音控股": "electronics",
    "汇川技术": "machinery",
}


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
    if text in STOCK_NAME_OVERRIDES:
        return STOCK_NAME_OVERRIDES[text]
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
        "ret_3",
        "ret_5",
        "ret_20",
        "short_momentum_3_5",
        "trend_alignment_5_20",
        "volume_breakout_3",
        "volume_breakout_5",
        "breakout_strength_20",
        "breakout_volume_confirm_20",
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

    if {"ret_3", "ret_5", "volume_ratio_5"}.issubset(out.columns):
        industry_strength = grouped["ret_3"].transform("mean") + grouped["ret_5"].transform("mean")
        volume_confirmation = grouped["volume_ratio_5"].transform("mean").clip(lower=0.0)
        feature_blocks["industry_collective_momentum"] = industry_strength
        feature_blocks["industry_collective_volume_confirm"] = industry_strength * volume_confirmation
        feature_blocks["theme_event_pressure"] = (
            out["short_momentum_3_5"].fillna(0.0)
            * feature_blocks["industry_collective_momentum"].fillna(0.0)
            * (1.0 + volume_confirmation.fillna(0.0))
        )

    if feature_blocks:
        out = pd.concat([out, pd.DataFrame(feature_blocks, index=out.index)], axis=1)

    return out
