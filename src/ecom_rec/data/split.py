"""按时间戳划分训练集/验证集/测试集，严格防止数据穿越"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from omegaconf import DictConfig

from ecom_rec.utils.logger import get_logger
from ecom_rec.utils.io import read_parquet, write_parquet

log = get_logger(__name__)


def split_data(cfg: DictConfig) -> None:
    """
    划分策略：
    1. 全局按时间戳排序，取最后 10% 时间段作为测试集（time-based split）
    2. 在训练+验证集内，取各用户最后一次交互作为验证集（LOO valid）
    3. 其余为训练集
    确保：train < valid < test（时间上单调不重叠）
    """
    interim = Path(cfg.interim_dir)
    processed = Path(cfg.processed_dir)

    df = read_parquet(interim / "interactions.parquet")
    df = df.sort("timestamp_sec")

    n = len(df)
    test_start_idx = int(n * (cfg.train_ratio + cfg.valid_ratio))
    # 用时间阈值切分（比按行索引更严格）
    timestamps = df["timestamp_sec"].to_list()
    test_threshold = timestamps[test_start_idx]
    valid_threshold_idx = int(n * cfg.train_ratio)
    valid_threshold = timestamps[valid_threshold_idx]

    test_df = df.filter(pl.col("timestamp_sec") >= test_threshold)
    train_valid_df = df.filter(pl.col("timestamp_sec") < test_threshold)

    # 在 train_valid 中做 LOO：每用户最后一次 → valid
    train_valid_sorted = train_valid_df.sort(["user_id", "timestamp_sec"])
    row_num = train_valid_sorted.with_row_index("_row")
    last_per_user = (
        row_num.group_by("user_id")
        .agg(pl.col("_row").max().alias("last_row"))
    )
    last_rows = set(last_per_user["last_row"].to_list())
    is_valid = row_num["_row"].map_elements(lambda r: r in last_rows, return_dtype=pl.Boolean)
    valid_df = train_valid_sorted.filter(is_valid)
    train_df = train_valid_sorted.filter(~is_valid)

    # 仅保留 test/valid 中 train 内出现过的用户（冷启动用户不参与评估）
    train_users = set(train_df["user_id"].to_list())
    valid_df = valid_df.filter(pl.col("user_id").is_in(train_users))
    test_df = test_df.filter(pl.col("user_id").is_in(train_users))

    log.info(
        f"划分完成：train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}"
    )
    log.info(
        f"用户数：train={train_df['user_id'].n_unique():,}  "
        f"valid={valid_df['user_id'].n_unique():,}  "
        f"test={test_df['user_id'].n_unique():,}"
    )

    write_parquet(train_df, processed / "train.parquet")
    write_parquet(valid_df, processed / "valid.parquet")
    write_parquet(test_df, processed / "test.parquet")

    # 保存用户/商品 ID 映射（供后续 embedding 使用）
    all_users = sorted(df["user_id"].unique().to_list())
    all_items = sorted(df["item_id"].unique().to_list())
    user_map = pl.DataFrame({"user_id": all_users, "user_idx": list(range(len(all_users)))})
    item_map = pl.DataFrame({"item_id": all_items, "item_idx": list(range(len(all_items)))})
    write_parquet(user_map, processed / "user_map.parquet")
    write_parquet(item_map, processed / "item_map.parquet")
    log.info(f"用户数：{len(all_users):,}，商品数：{len(all_items):,}")
