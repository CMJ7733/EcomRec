"""清洗 Amazon Reviews 2023 原始数据并进行 K-core 过滤"""
from __future__ import annotations

import json
import gzip
from pathlib import Path

import polars as pl
from omegaconf import DictConfig

from ecom_rec.utils.logger import get_logger
from ecom_rec.utils.io import write_parquet

log = get_logger(__name__)


def _load_reviews(gz_path: Path) -> pl.DataFrame:
    """读取 jsonl.gz 评论文件，仅保留必要字段"""
    records = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            records.append({
                "user_id": obj.get("user_id", ""),
                "item_id": obj.get("asin", ""),
                "rating": float(obj.get("rating", 0.0)),
                "timestamp": int(obj.get("timestamp", 0)),
            })
    df = pl.DataFrame(records)
    log.info(f"加载评论：{len(df):,} 条")
    return df


def _load_meta(gz_path: Path) -> pl.DataFrame:
    """读取 jsonl.gz 元数据，仅保留关键字段"""
    records = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # 安全提取 category（可能是嵌套列表）
            cats = obj.get("categories", [])
            category = cats[0] if cats else ""
            if isinstance(category, list):
                category = category[0] if category else ""
            price_raw = obj.get("price", None)
            try:
                price = float(str(price_raw).replace("$", "").replace(",", "")) if price_raw else None
            except ValueError:
                price = None
            records.append({
                "item_id": obj.get("parent_asin", obj.get("asin", "")),
                "title": str(obj.get("title", ""))[:200],
                "brand": str(obj.get("brand", "")),
                "category": str(category),
                "price": price,
            })
    df = pl.DataFrame(records).unique(subset=["item_id"])
    log.info(f"加载元数据：{len(df):,} 个商品")
    return df


def _kcore_filter(df: pl.DataFrame, k: int = 5) -> pl.DataFrame:
    """迭代过滤：剔除交互次数 < k 的用户和商品，直到收敛"""
    prev_len = len(df) + 1
    iteration = 0
    while len(df) < prev_len:
        prev_len = len(df)
        item_counts = df.group_by("item_id").agg(pl.len().alias("cnt"))
        df = df.join(item_counts.filter(pl.col("cnt") >= k).select("item_id"), on="item_id")
        user_counts = df.group_by("user_id").agg(pl.len().alias("cnt"))
        df = df.join(user_counts.filter(pl.col("cnt") >= k).select("user_id"), on="user_id")
        iteration += 1
        log.info(f"K-core 第 {iteration} 轮：{len(df):,} 条交互")
    log.info(f"K-core({k}) 过滤完成：{len(df):,} 条，{df['user_id'].n_unique():,} 用户，{df['item_id'].n_unique():,} 商品")
    return df


def clean_data(cfg: DictConfig) -> None:
    """主清洗流程：加载 → 去重 → K-core → 解析时间特征 → 合并元数据 → 落盘"""
    raw = Path(cfg.raw_dir)
    interim = Path(cfg.interim_dir)

    # 1. 加载评论
    review_gz = raw / "Beauty_and_Personal_Care.jsonl.gz"
    meta_gz = raw / "meta_Beauty_and_Personal_Care.jsonl.gz"
    reviews = _load_reviews(review_gz)

    # 2. 去除完全重复行、去除空 user_id/item_id
    reviews = reviews.filter(
        (pl.col("user_id") != "") & (pl.col("item_id") != "")
    ).unique(subset=["user_id", "item_id", "timestamp"])
    log.info(f"去重后：{len(reviews):,} 条")

    # 3. K-core 过滤
    reviews = _kcore_filter(reviews, k=cfg.k_core)

    # 4. 解析时间特征（timestamp 为毫秒或秒，自动适配）
    # Amazon 2023 timestamp 以毫秒为单位
    reviews = reviews.with_columns([
        (pl.col("timestamp") // 1000).alias("timestamp_sec"),
    ]).with_columns([
        pl.from_epoch("timestamp_sec", time_unit="s").alias("datetime"),
    ]).with_columns([
        pl.col("datetime").dt.year().alias("year"),
        pl.col("datetime").dt.month().alias("month"),
        pl.col("datetime").dt.weekday().alias("weekday"),
        pl.col("datetime").dt.hour().alias("hour"),
    ]).drop("datetime")

    # 5. 加载并合并元数据
    meta = _load_meta(meta_gz)
    reviews = reviews.join(meta, on="item_id", how="left")

    # 6. 落盘
    write_parquet(reviews, interim / "interactions.parquet")
    log.info(f"清洗完成，已写入 {interim / 'interactions.parquet'}")
