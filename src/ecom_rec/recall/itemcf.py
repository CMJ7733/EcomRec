"""Item-based 协同过滤召回：共现矩阵 + IUF 衰减"""
from __future__ import annotations

import numpy as np
import polars as pl
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as sparse_norm

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
            pairs = sorted(zip(row["ts"], row["items"]))
            user_items[uid] = [item for _, item in pairs]
        self._user_history = user_items

        # 统计商品热度
        item_pop: dict[str, int] = defaultdict(int)
        for items in user_items.values():
            for item in items:
                item_pop[item] += 1
        self._item_pop = dict(item_pop)

        # 向量化构建共现矩阵（稀疏矩阵乘法）
        all_items = sorted(item_pop.keys())
        item_to_idx = {item: idx for idx, item in enumerate(all_items)}
        n_items = len(all_items)

        # 构建稀疏用户-商品矩阵（带 IUF 权重）
        uid_list = list(user_items.keys())
        rows, cols, vals = [], [], []
        for row_idx, (uid, items) in enumerate(user_items.items()):
            iuf_weight = 1.0 / np.log1p(len(items)) if self.use_iuf else 1.0
            seen = set()
            for item in items:
                if item not in seen:
                    cols.append(item_to_idx[item])
                    rows.append(row_idx)
                    vals.append(iuf_weight)
                    seen.add(item)

        X = csr_matrix((vals, (rows, cols)), shape=(len(uid_list), n_items))
        log.info(f"ItemCF 稀疏矩阵构建完成：{X.shape}，计算共现矩阵 ...")

        # 稀疏共现矩阵 X^T @ X，保持稀疏
        co_occur_sparse = X.T @ X

        # 余弦归一化：对角线开根号
        diag = np.sqrt(np.asarray(co_occur_sparse.diagonal()).flatten())
        diag[diag == 0] = 1.0

        # 提取 top-K 近邻（逐行处理稀疏矩阵）
        item_sim: dict[str, list[tuple[str, float]]] = {}
        for i in range(n_items):
            row_start = co_occur_sparse.indptr[i]
            row_end = co_occur_sparse.indptr[i + 1]
            if row_start == row_end:
                continue
            indices = co_occur_sparse.indices[row_start:row_end]
            data = co_occur_sparse.data[row_start:row_end].copy()
            if len(data) == 0:
                continue

            # 余弦归一化
            norm_i = diag[i]
            data /= (norm_i * diag[indices] + 1e-9)

            # 排除自身
            mask = indices != i
            indices = indices[mask]
            data = data[mask]

            if len(data) == 0:
                continue

            # top-K
            k = min(self.n_neighbors, len(data))
            if k < len(data):
                top_k_idx = np.argpartition(data, -k)[-k:]
            else:
                top_k_idx = np.arange(len(data))
            top_k_idx = top_k_idx[np.argsort(data[top_k_idx])[::-1]]
            item_sim[all_items[i]] = [(all_items[indices[j]], float(data[j])) for j in top_k_idx]

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
