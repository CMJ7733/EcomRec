"""多路召回融合：加权合并多个召回器的候选集"""
from __future__ import annotations

from collections import defaultdict

import polars as pl

from ecom_rec.recall.base import Recaller
from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)


class MultiRecall:
    """
    多路召回融合器。

    将多个 Recaller 的结果按权重加权合并，去重后输出候选集。
    权重越高的召回器，其结果在最终排名中权重越大。
    """

    def __init__(self, recallers: list[tuple[Recaller, float]]) -> None:
        """
        Args:
            recallers: [(recaller_instance, weight), ...]，weight 相对大小决定权重占比
        """
        self.recallers = recallers
        total_w = sum(w for _, w in recallers)
        self._weights = [w / total_w for _, w in recallers]

    def recommend(self, user_id: str, k: int = 200) -> list[str]:
        """
        融合多路召回结果。

        策略：每路召回器各取 k 候选，按位置赋予衰减分数（1/rank），
        再乘以该路的权重，最终按总分降序返回 Top-k。
        """
        scores: dict[str, float] = defaultdict(float)
        for (recaller, _), weight in zip(self.recallers, self._weights):
            candidates = recaller.recommend(user_id, k=k)
            for rank, item_id in enumerate(candidates):
                # 位置衰减分数：第 1 名得 1/(1+0)=1，第 2 名得 1/2，...
                scores[item_id] += weight * (1.0 / (rank + 1))

        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        return [iid for iid, _ in sorted_items[:k]]

    def recommend_batch(self, user_ids: list[str], k: int = 200) -> dict[str, list[str]]:
        """批量融合多路召回结果"""
        return {uid: self.recommend(uid, k) for uid in user_ids}

    # ── Backward-compatible aliases ──────────────────────────────────────────
    def retrieve(self, user_id: str, k: int = 200, k_per_recaller: int = 100) -> list[str]:
        """retrieve 是 recommend 的别名（向后兼容）"""
        return self.recommend(user_id, k=k)

    def retrieve_batch(
        self,
        user_ids: list[str],
        k: int = 200,
        k_per_recaller: int = 100,
    ) -> dict[str, list[str]]:
        """retrieve_batch 是 recommend_batch 的别名（向后兼容）"""
        return self.recommend_batch(user_ids, k=k)
