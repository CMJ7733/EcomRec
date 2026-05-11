"""BPR-MF 召回：贝叶斯个性化排序矩阵分解（使用 implicit 库）"""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp

from ecom_rec.recall.base import Recaller
from ecom_rec.utils.logger import get_logger
from ecom_rec.utils.seed import set_seed

log = get_logger(__name__)


class BPRRecaller(Recaller):
    """使用 implicit 库的 BPR-MF 召回模型。"""

    def __init__(
        self,
        factors: int = 64,
        iterations: int = 100,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.random_state = random_state
        self._model = None
        self._user_idx: dict[str, int] = {}
        self._item_idx: dict[str, int] = {}
        self._idx_item: dict[int, str] = {}
        self._user_history: dict[str, set[str]] = {}

    def fit(self, interactions: pl.DataFrame) -> "BPRRecaller":
        from implicit.bpr import BayesianPersonalizedRanking

        set_seed(self.random_state)

        # 构建 ID 映射
        users = sorted(interactions["user_id"].unique().to_list())
        items = sorted(interactions["item_id"].unique().to_list())
        self._user_idx = {u: i for i, u in enumerate(users)}
        self._item_idx = {it: i for i, it in enumerate(items)}
        self._idx_item = {i: it for it, i in self._item_idx.items()}

        # 构建 user-item 稀疏矩阵（使用评分作为置信度）
        rows = interactions["user_id"].map_elements(
            lambda u: self._user_idx[u], return_dtype=pl.Int32
        ).to_list()
        cols = interactions["item_id"].map_elements(
            lambda it: self._item_idx[it], return_dtype=pl.Int32
        ).to_list()
        data = interactions["rating"].to_list()
        user_item = sp.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

        # 保存用户历史
        for row in interactions.group_by("user_id").agg(
            pl.col("item_id").alias("items")
        ).iter_rows(named=True):
            self._user_history[row["user_id"]] = set(row["items"])

        self._model = BayesianPersonalizedRanking(
            factors=self.factors,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            random_state=self.random_state,
        )
        self._model.fit(user_item)
        log.info(f"BPRRecaller 训练完成：{len(users):,} 用户 × {len(items):,} 商品")
        return self

    def recommend(self, user_id: str, k: int = 50) -> list[str]:
        if user_id not in self._user_idx:
            return []
        uid = self._user_idx[user_id]
        history = self._user_history.get(user_id, set())
        filter_items = [self._item_idx[it] for it in history if it in self._item_idx]

        item_ids, _ = self._model.recommend(
            uid, None, N=k + len(filter_items), filter_already_liked_items=True
        )
        return [self._idx_item[i] for i in item_ids if i in self._idx_item][:k]
