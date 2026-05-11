"""Item-based 协同过滤召回：共现矩阵 + IUF 衰减"""
from __future__ import annotations

import numpy as np
import polars as pl
from collections import defaultdict

from ecom_rec.recall.base import Recaller
from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)


class ItemCFRecaller(Recaller):
    """基于 Item-Item 协同过滤的召回。

    IUF（Inverse User Frequency）衰减：活跃用户对物品相似度的贡献
    被其活跃度的对数惩罚，降低超级活跃用户对热门物品相似度的虚假拉升。
    """

    def __init__(self, n_neighbors: int = 20, use_iuf: bool = True) -> None:
        self.n_neighbors = n_neighbors
        self.use_iuf = use_iuf
        self._item_sim: dict[str, list[tuple[str, float]]] = {}
        self._user_history: dict[str, list[str]] = {}
        self._item_pop: dict[str, int] = {}

    def fit(self, interactions: pl.DataFrame) -> "ItemCFRecaller":
        # 构建用户-商品交互字典
        user_items: dict[str, list[str]] = {}
        for row in interactions.group_by("user_id").agg(
            pl.col("item_id").alias("items"),
            pl.col("timestamp_sec").alias("ts")
        ).iter_rows(named=True):
            uid = row["user_id"]
            # 按时间排序（最近的放后面）
            pairs = sorted(zip(row["ts"], row["items"]))
            user_items[uid] = [item for _, item in pairs]
        self._user_history = user_items

        # 统计商品热度（用于 IUF）
        item_pop: dict[str, int] = defaultdict(int)
        for items in user_items.values():
            for item in items:
                item_pop[item] += 1
        self._item_pop = dict(item_pop)

        # 构建商品共现矩阵
        co_occur: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for uid, items in user_items.items():
            # IUF 权重：活跃用户贡献降低
            iuf_weight = 1.0 / np.log1p(len(items)) if self.use_iuf else 1.0
            item_set = list(set(items))
            for i in range(len(item_set)):
                for j in range(i + 1, len(item_set)):
                    a, b = item_set[i], item_set[j]
                    co_occur[a][b] += iuf_weight
                    co_occur[b][a] += iuf_weight

        # 归一化（余弦相似度）
        item_sim: dict[str, list[tuple[str, float]]] = {}
        for item_a, neighbors in co_occur.items():
            norm_a = np.sqrt(item_pop.get(item_a, 1))
            sims = []
            for item_b, cnt in neighbors.items():
                norm_b = np.sqrt(item_pop.get(item_b, 1))
                sim = cnt / (norm_a * norm_b + 1e-9)
                sims.append((item_b, sim))
            sims.sort(key=lambda x: -x[1])
            item_sim[item_a] = sims[:self.n_neighbors]

        self._item_sim = item_sim
        log.info(f"ItemCF 训练完成：{len(item_sim):,} 个商品有相似列表")
        return self

    def recommend(self, user_id: str, k: int = 50) -> list[str]:
        history = self._user_history.get(user_id, [])
        history_set = set(history)

        # 对用户历史商品的相似商品加权求和
        scores: dict[str, float] = defaultdict(float)
        # 给最近交互的商品更高权重（位置权重衰减）
        n = len(history)
        for idx, item in enumerate(history):
            position_weight = (idx + 1) / n  # 越近权重越高
            for sim_item, sim_score in self._item_sim.get(item, []):
                if sim_item not in history_set:
                    scores[sim_item] += sim_score * position_weight

        candidates = sorted(scores.items(), key=lambda x: -x[1])
        return [iid for iid, _ in candidates[:k]]
