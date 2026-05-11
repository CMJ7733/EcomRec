"""Top-Pop 全局热门召回：兜底策略"""
from __future__ import annotations

import polars as pl

from ecom_rec.recall.base import Recaller
from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)


class PopRecaller(Recaller):
    """基于全局热门度（交互次数）的召回。

    策略：对所有用户返回相同的 Top-K 热门商品列表，
    并过滤掉用户已交互过的商品。
    """

    def __init__(self) -> None:
        self._top_items: list[str] = []
        self._user_history: dict[str, set[str]] = {}

    def fit(self, interactions: pl.DataFrame) -> "PopRecaller":
        # 按商品热度排序
        item_pop = (
            interactions.group_by("item_id")
            .agg(pl.len().alias("pop"))
            .sort("pop", descending=True)
        )
        self._top_items = item_pop["item_id"].to_list()

        # 记录每个用户的历史交互（用于过滤）
        for row in interactions.group_by("user_id").agg(pl.col("item_id").alias("items")).iter_rows(named=True):
            self._user_history[row["user_id"]] = set(row["items"])

        log.info(f"PopRecaller 训练完成：{len(self._top_items):,} 个候选商品")
        return self

    def recommend(self, user_id: str, k: int = 50) -> list[str]:
        history = self._user_history.get(user_id, set())
        candidates = [iid for iid in self._top_items if iid not in history]
        return candidates[:k]
