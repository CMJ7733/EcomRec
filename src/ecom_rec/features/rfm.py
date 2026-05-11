"""计算 RFM 用户指标

R (Recency): 距参考日期的天数（越小越好）
F (Frequency): 历史交互次数
M (Monetary): 使用 sum(rating * price) 近似（Amazon 无真实消费金额）
              对于 price 缺失的记录，用类目均价填充后计算
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)

REFERENCE_DATE_UNIX = int(datetime(2024, 1, 1).timestamp())  # 参考日期：数据集截止点


def compute_rfm(df: pl.DataFrame, reference_ts: int = REFERENCE_DATE_UNIX) -> pl.DataFrame:
    """
    计算用户 RFM 指标。

    Args:
        df: 含 user_id, timestamp_sec, rating, price 字段的交互 DataFrame
        reference_ts: 参考时间戳（秒），默认 2024-01-01

    Returns:
        DataFrame，列为 [user_id, recency_days, frequency, monetary, last_ts]
    """
    # 处理 price 缺失：先用类目均价填充，再用全局均价兜底
    if "price" in df.columns and "category" in df.columns:
        category_avg_price = (
            df.filter(pl.col("price").is_not_null())
            .group_by("category")
            .agg(pl.col("price").mean().alias("cat_avg_price"))
        )
        df = df.join(category_avg_price, on="category", how="left")
        global_avg_price = df.filter(pl.col("price").is_not_null())["price"].mean() or 1.0
        df = df.with_columns(
            pl.col("price")
            .fill_null(pl.col("cat_avg_price"))
            .fill_null(global_avg_price)
            .alias("price_filled")
        )
        monetary_col = (pl.col("rating") * pl.col("price_filled")).alias("value")
        df = df.with_columns(monetary_col)
        monetary_agg = pl.col("value").sum().alias("monetary")
    else:
        # 无价格信息，用评分之和近似
        df = df.with_columns(pl.col("rating").alias("value"))
        monetary_agg = pl.col("value").sum().alias("monetary")

    rfm = df.group_by("user_id").agg([
        ((reference_ts - pl.col("timestamp_sec").max()) / 86400).cast(pl.Float64).alias("recency_days"),
        pl.len().alias("frequency"),
        monetary_agg,
        pl.col("timestamp_sec").max().alias("last_ts"),
    ])

    log.info(f"RFM 计算完成：{len(rfm):,} 个用户")
    return rfm
