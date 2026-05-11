"""
CTR 精排特征工程

特征分为三类：
1. 稠密特征（Dense）：连续数值，归一化到 [0, 1]
   - user_avg_rating: 用户历史平均评分
   - user_active_days: 用户活跃天数（首次到最后交互）
   - user_frequency: 用户交互总次数
   - item_avg_rating: 商品平均评分
   - item_review_count: 商品评论数
   - item_price_quantile: 价格在全量商品中的分位数

2. 稀疏特征（Sparse，返回整数 Index，用于 Embedding 查表）
   - user_idx: 用户 ID 映射后的整数索引
   - item_idx: 商品 ID 映射后的整数索引
   - category_idx: 商品类目整数索引
   - brand_idx: 商品品牌整数索引
   - weekday: 0-6（星期几）
   - hour: 0-23（小时）

3. 标签
   - label: 1（正样本：真实交互）/ 0（负样本：随机采样）
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from ecom_rec.utils.logger import get_logger
from ecom_rec.utils.io import read_parquet, write_parquet
from ecom_rec.utils.seed import set_seed

log = get_logger(__name__)


class FeatureSpec(NamedTuple):
    """特征元数据，供模型层使用"""
    dense_features: list[str]
    sparse_features: list[str]
    sparse_vocab_sizes: dict[str, int]  # 每个稀疏特征的词汇表大小


def build_ctr_features(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    test: pl.DataFrame,
    user_map: pl.DataFrame,
    item_map: pl.DataFrame,
    neg_sample_ratio: int = 4,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, FeatureSpec]:
    """
    构建 CTR 特征并进行负采样。

    Args:
        train/valid/test: 正样本交互数据
        user_map/item_map: ID → 整数索引映射
        neg_sample_ratio: 每个正样本对应的负样本数
        random_state: 随机种子

    Returns:
        (train_feat, valid_feat, test_feat, feature_spec)
        每个 DataFrame 含稠密特征 + 稀疏特征 + label 列
    """
    set_seed(random_state)

    # ---- 1. 计算统计特征（仅从 train 计算，避免数据穿越）----

    # 用户统计
    user_stats = train.group_by("user_id").agg([
        pl.col("rating").mean().alias("user_avg_rating"),
        pl.len().alias("user_frequency"),
        ((pl.col("timestamp_sec").max() - pl.col("timestamp_sec").min()) / 86400.0)
        .alias("user_active_days"),
    ])

    # 商品统计
    item_stats = train.group_by("item_id").agg([
        pl.col("rating").mean().alias("item_avg_rating"),
        pl.len().alias("item_review_count"),
    ])

    # 价格分位数（用 train 中有价格的商品）
    if "price" in train.columns:
        prices_valid = train.filter(pl.col("price").is_not_null())["price"].to_numpy()
        price_quantiles = np.percentile(prices_valid, np.arange(0, 101, 1)) if len(prices_valid) > 0 else None
    else:
        price_quantiles = None

    def price_to_quantile(p):
        if p is None or price_quantiles is None:
            return 0.5
        return float(np.searchsorted(price_quantiles, p) / 100.0)

    # ---- 2. 构建类目/品牌 ID 映射 ----
    all_interactions = pl.concat([train, valid, test])

    if "category" in train.columns:
        categories = sorted(all_interactions["category"].drop_nulls().unique().to_list())
        cat_map = {c: i + 1 for i, c in enumerate(categories)}  # 0 保留给未知
        cat_map[""] = 0
    else:
        cat_map = {}

    if "brand" in train.columns:
        brands = sorted(all_interactions["brand"].drop_nulls().unique().to_list())
        brand_map = {b: i + 1 for i, b in enumerate(brands)}
        brand_map[""] = 0
    else:
        brand_map = {}

    # user/item 映射转为字典
    uid_map = {row[0]: row[1] for row in user_map.iter_rows()}
    iid_map = {row[0]: row[1] for row in item_map.iter_rows()}
    all_item_ids = list(iid_map.keys())

    DENSE_FEATURES = [
        "user_avg_rating", "user_frequency", "user_active_days",
        "item_avg_rating", "item_review_count", "item_price_quantile",
    ]
    SPARSE_FEATURES = ["user_idx", "item_idx", "category_idx", "brand_idx", "weekday", "hour"]

    def _select_feature_cols(df: pl.DataFrame) -> pl.DataFrame:
        needed_cols = DENSE_FEATURES + SPARSE_FEATURES + ["label"]
        # 补充缺失的列为 0
        for c in needed_cols:
            if c not in df.columns:
                df = df.with_columns(pl.lit(0).alias(c))
        return df.select(needed_cols)

    # ---- 3. 特征构建函数 ----
    def build_features_for_split(df: pl.DataFrame, include_neg: bool = True) -> pl.DataFrame:
        # 正样本：合并统计特征
        pos = (
            df
            .join(user_stats, on="user_id", how="left")
            .join(item_stats, on="item_id", how="left")
            .join(user_map, on="user_id", how="left")
            .join(item_map, on="item_id", how="left")
        )

        # 价格分位
        if "price" in pos.columns and price_quantiles is not None:
            pos = pos.with_columns(
                pl.col("price").map_elements(price_to_quantile, return_dtype=pl.Float64)
                .alias("item_price_quantile")
            )
        else:
            pos = pos.with_columns(pl.lit(0.5).alias("item_price_quantile"))

        # 类目 / 品牌索引
        if "category" in pos.columns:
            pos = pos.with_columns(
                pl.col("category").fill_null("").map_elements(
                    lambda c: cat_map.get(str(c), 0), return_dtype=pl.Int32
                ).alias("category_idx")
            )
        else:
            pos = pos.with_columns(pl.lit(0).cast(pl.Int32).alias("category_idx"))

        if "brand" in pos.columns:
            pos = pos.with_columns(
                pl.col("brand").fill_null("").map_elements(
                    lambda b: brand_map.get(str(b), 0), return_dtype=pl.Int32
                ).alias("brand_idx")
            )
        else:
            pos = pos.with_columns(pl.lit(0).cast(pl.Int32).alias("brand_idx"))

        pos = pos.with_columns(pl.lit(1).cast(pl.Int8).alias("label"))

        # 填充 NaN（冷启动用户/商品的统计特征）
        pos = pos.with_columns([
            pl.col("user_avg_rating").fill_null(4.0),
            pl.col("user_frequency").fill_null(5),
            pl.col("user_active_days").fill_null(0.0),
            pl.col("item_avg_rating").fill_null(4.0),
            pl.col("item_review_count").fill_null(1),
            pl.col("user_idx").fill_null(0),
            pl.col("item_idx").fill_null(0),
        ])

        if not include_neg:
            return _select_feature_cols(pos)

        # ---- 负采样 ----
        # 构建每个用户的历史交互集合（用于过滤正样本）
        user_histories: dict[str, set] = {}
        for row in df.group_by("user_id").agg(pl.col("item_id").alias("items")).iter_rows(named=True):
            items = row["items"]
            if hasattr(items, "to_list"):
                user_histories[row["user_id"]] = set(items.to_list())
            else:
                user_histories[row["user_id"]] = set(items)

        neg_records = []
        for row in df.iter_rows(named=True):
            uid = row["user_id"]
            history = user_histories.get(uid, set())
            sampled = 0
            attempts = 0
            while sampled < neg_sample_ratio and attempts < neg_sample_ratio * 20:
                neg_item = random.choice(all_item_ids)
                if neg_item not in history:
                    neg_records.append({
                        "user_id": uid,
                        "item_id": neg_item,
                        "rating": 0.0,
                        "timestamp_sec": row.get("timestamp_sec", 0),
                        "weekday": row.get("weekday", 0),
                        "hour": row.get("hour", 0),
                    })
                    sampled += 1
                attempts += 1

        if neg_records:
            neg_df = pl.DataFrame(neg_records)
            neg_feat = (
                neg_df
                .join(user_stats, on="user_id", how="left")
                .join(item_stats, on="item_id", how="left")
                .join(user_map, on="user_id", how="left")
                .join(item_map, on="item_id", how="left")
            )
            # 为负样本添加缺失列
            if "category" not in neg_feat.columns:
                neg_feat = neg_feat.with_columns(pl.lit("").alias("category"))
            if "brand" not in neg_feat.columns:
                neg_feat = neg_feat.with_columns(pl.lit("").alias("brand"))
            if "price" not in neg_feat.columns:
                neg_feat = neg_feat.with_columns(pl.lit(None).cast(pl.Float64).alias("price"))

            neg_feat = neg_feat.with_columns([
                pl.col("user_avg_rating").fill_null(4.0),
                pl.col("user_frequency").fill_null(5),
                pl.col("user_active_days").fill_null(0.0),
                pl.col("item_avg_rating").fill_null(4.0),
                pl.col("item_review_count").fill_null(1),
                pl.col("user_idx").fill_null(0),
                pl.col("item_idx").fill_null(0),
                pl.lit(0.5).alias("item_price_quantile"),
                pl.lit(0).cast(pl.Int32).alias("category_idx"),
                pl.lit(0).cast(pl.Int32).alias("brand_idx"),
                pl.lit(0).cast(pl.Int8).alias("label"),
            ])
            result = pl.concat([_select_feature_cols(pos), _select_feature_cols(neg_feat)])
        else:
            result = _select_feature_cols(pos)

        return result.sample(fraction=1.0, shuffle=True, seed=random_state)

    log.info("构建训练集特征（含负采样）...")
    train_feat = build_features_for_split(train, include_neg=True)
    log.info(
        f"训练集：{len(train_feat):,} 条"
        f"（正:{train_feat['label'].sum():,} 负:{len(train_feat) - train_feat['label'].sum():,}）"
    )

    log.info("构建验证集特征（不做额外负采样，使用原始正样本）...")
    valid_feat = build_features_for_split(valid, include_neg=False)

    log.info("构建测试集特征...")
    test_feat = build_features_for_split(test, include_neg=False)

    spec = FeatureSpec(
        dense_features=DENSE_FEATURES,
        sparse_features=SPARSE_FEATURES,
        sparse_vocab_sizes={
            "user_idx": len(uid_map) + 1,
            "item_idx": len(iid_map) + 1,
            "category_idx": len(cat_map) + 1,
            "brand_idx": len(brand_map) + 1,
            "weekday": 7,
            "hour": 24,
        },
    )
    return train_feat, valid_feat, test_feat, spec


def save_ctr_features(
    train_feat: pl.DataFrame,
    valid_feat: pl.DataFrame,
    test_feat: pl.DataFrame,
    spec: FeatureSpec,
    processed_dir: str = "data/processed",
) -> None:
    """将特征 DataFrame 落盘到 processed/"""
    p = Path(processed_dir)
    write_parquet(train_feat, p / "ctr_train.parquet")
    write_parquet(valid_feat, p / "ctr_valid.parquet")
    write_parquet(test_feat, p / "ctr_test.parquet")

    # 保存 FeatureSpec 为 JSON
    spec_dict = {
        "dense_features": spec.dense_features,
        "sparse_features": spec.sparse_features,
        "sparse_vocab_sizes": spec.sparse_vocab_sizes,
    }
    (p / "feature_spec.json").write_text(json.dumps(spec_dict, ensure_ascii=False, indent=2))
    log.info(f"CTR 特征已落盘到 {p}")
