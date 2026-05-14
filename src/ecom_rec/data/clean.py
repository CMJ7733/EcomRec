"""清洗 Amazon Reviews 2023 原始数据并进行 K-core 过滤"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
from omegaconf import DictConfig

from ecom_rec.utils.logger import get_logger
from ecom_rec.utils.io import write_parquet

log = get_logger(__name__)


def _load_reviews(gz_path: Path, chunk_size: int = 1_000_000) -> pl.DataFrame:
    """流式分块读取 jsonl.gz 评论文件，仅保留 4 个字段，避免一次性载入内存"""
    user_ids: list[str] = []
    item_ids: list[str] = []
    ratings: list[float] = []
    timestamps: list[int] = []
    chunks: list[pl.DataFrame] = []
    schema = {
        "user_id": pl.Utf8,
        "item_id": pl.Utf8,
        "rating": pl.Float64,
        "timestamp": pl.Int64,
    }

    def _flush() -> None:
        if not user_ids:
            return
        chunks.append(pl.DataFrame(
            {"user_id": user_ids, "item_id": item_ids, "rating": ratings, "timestamp": timestamps},
            schema=schema,
        ))
        user_ids.clear()
        item_ids.clear()
        ratings.clear()
        timestamps.clear()

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            user_ids.append(obj.get("user_id") or "")
            item_ids.append(obj.get("asin") or "")
            try:
                ratings.append(float(obj.get("rating") or 0.0))
            except (TypeError, ValueError):
                ratings.append(0.0)
            try:
                timestamps.append(int(obj.get("timestamp") or 0))
            except (TypeError, ValueError):
                timestamps.append(0)
            if len(user_ids) >= chunk_size:
                _flush()
                log.info(f"已加载 {i:,} 条评论...")

    _flush()
    df = pl.concat(chunks) if chunks else pl.DataFrame(schema=schema)
    log.info(f"加载评论：{len(df):,} 条")
    return df


def _load_meta(gz_path: Path, chunk_size: int = 200_000) -> pl.DataFrame:
    """流式分块读取 jsonl.gz 元数据，仅保留 5 个字段"""
    item_ids: list[str] = []
    titles: list[str] = []
    brands: list[str] = []
    categories: list[str] = []
    prices: list[float | None] = []
    chunks: list[pl.DataFrame] = []
    schema = {
        "item_id": pl.Utf8,
        "title": pl.Utf8,
        "brand": pl.Utf8,
        "category": pl.Utf8,
        "price": pl.Float64,
    }

    def _flush() -> None:
        if not item_ids:
            return
        chunks.append(pl.DataFrame(
            {"item_id": item_ids, "title": titles, "brand": brands, "category": categories, "price": prices},
            schema=schema,
        ))
        item_ids.clear()
        titles.clear()
        brands.clear()
        categories.clear()
        prices.clear()

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)

            # category: 优先 main_category，否则取 categories 列表第一项
            cat = obj.get("main_category")
            if not cat:
                cats = obj.get("categories") or []
                cat = cats[0] if cats else ""
                if isinstance(cat, list):
                    cat = cat[0] if cat else ""

            # price 清洗
            price_raw = obj.get("price")
            price: float | None
            try:
                price = float(str(price_raw).replace("$", "").replace(",", "")) if price_raw else None
            except (TypeError, ValueError):
                price = None

            item_ids.append(obj.get("parent_asin") or obj.get("asin") or "")
            titles.append(str(obj.get("title") or "")[:200])
            brands.append(str(obj.get("brand") or obj.get("store") or ""))
            categories.append(str(cat or ""))
            prices.append(price)

            if len(item_ids) >= chunk_size:
                _flush()
                log.info(f"已加载 {i:,} 个商品...")

    _flush()
    df = pl.concat(chunks) if chunks else pl.DataFrame(schema=schema)
    df = df.unique(subset=["item_id"])
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
