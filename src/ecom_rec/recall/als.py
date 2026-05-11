"""ALS 召回：交替最小二乘矩阵分解（使用 implicit 库）"""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp

from ecom_rec.recall.base import Recaller
from ecom_rec.utils.logger import get_logger
from ecom_rec.utils.seed import set_seed

log = get_logger(__name__)


class ALSRecaller(Recaller):
    """使用 implicit 库的 ALS 召回模型（GPU 可选）。"""

    def __init__(
        self,
        factors: int = 64,
        iterations: int = 50,
        regularization: float = 0.01,
        alpha: float = 40.0,
        random_state: int = 42,
        use_gpu: bool = False,
    ) -> None:
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha
        self.random_state = random_state
        self.use_gpu = use_gpu
        self._model = None
        self._user_idx: dict[str, int] = {}
        self._item_idx: dict[str, int] = {}
        self._idx_item: dict[int, str] = {}
        self._user_item_matrix = None
        self._user_history: dict[str, set[str]] = {}

    def fit(self, interactions: pl.DataFrame) -> "ALSRecaller":
        from implicit.als import AlternatingLeastSquares

        set_seed(self.random_state)

        users = sorted(interactions["user_id"].unique().to_list())
        items = sorted(interactions["item_id"].unique().to_list())
        self._user_idx = {u: i for i, u in enumerate(users)}
        self._item_idx = {it: i for i, it in enumerate(items)}
        self._idx_item = {i: it for it, i in self._item_idx.items()}

        rows = interactions["user_id"].map_elements(
            lambda u: self._user_idx[u], return_dtype=pl.Int32
        ).to_list()
        cols = interactions["item_id"].map_elements(
            lambda it: self._item_idx[it], return_dtype=pl.Int32
        ).to_list()
        data = [self.alpha] * len(rows)  # 置信度 = alpha（隐式反馈统一权重）

        user_item = sp.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        self._user_item_matrix = user_item

        for row in interactions.group_by("user_id").agg(
            pl.col("item_id").alias("items")
        ).iter_rows(named=True):
            self._user_history[row["user_id"]] = set(row["items"])

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.random_state,
            use_gpu=self.use_gpu,
        )
        self._model.fit(user_item)
        log.info(f"ALSRecaller 训练完成：{len(users):,} 用户 × {len(items):,} 商品")
        return self

    def recommend(self, user_id: str, k: int = 50) -> list[str]:
        if user_id not in self._user_idx:
            return []
        uid = self._user_idx[user_id]
        item_ids, _ = self._model.recommend(
            uid, self._user_item_matrix[uid], N=k, filter_already_liked_items=True
        )
        return [self._idx_item[i] for i in item_ids if i in self._idx_item][:k]
