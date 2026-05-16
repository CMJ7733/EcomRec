"""端到端推荐流水线：多路召回 → DeepFM 精排 → MMR 打散"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from ecom_rec.pipeline.multi_recall import MultiRecall
from ecom_rec.pipeline.rerank import mmr_rerank
from ecom_rec.utils.device import pick_device
from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)


class Recommender:
    """
    端到端推荐引擎。

    流程：
        1. 多路召回 → 200 候选
        2. CTR 特征构建 → DeepFM 精排 → Top-50
        3. MMR 打散 → Top-10
    """

    def __init__(
        self,
        multi_recall: MultiRecall,
        rank_model: Any,  # DeepFM 或 LGBRanker 等
        dense_features: list[str],
        sparse_features: list[str],
        user_stats: pl.DataFrame,
        item_stats: pl.DataFrame,
        user_map: pl.DataFrame,
        item_map: pl.DataFrame,
        item_meta: pl.DataFrame,
        model_type: str = "deepfm",
        device: str = "auto",
        recall_k: int = 200,
        rank_top_k: int = 50,
        final_k: int = 10,
        mmr_lambda: float = 0.5,
    ) -> None:
        self.multi_recall = multi_recall
        self.rank_model = rank_model
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.model_type = model_type
        self.recall_k = recall_k
        self.rank_top_k = rank_top_k
        self.final_k = final_k
        self.mmr_lambda = mmr_lambda

        # 预处理统计字典（加速推理时的特征 lookup）
        self._user_stats = {r["user_id"]: r for r in user_stats.iter_rows(named=True)}
        self._item_stats = {r["item_id"]: r for r in item_stats.iter_rows(named=True)}
        self._uid_map = {r["user_id"]: r["user_idx"] for r in user_map.iter_rows(named=True)}
        self._iid_map = {r["item_id"]: r["item_idx"] for r in item_map.iter_rows(named=True)}
        self._item_meta = {r["item_id"]: r for r in item_meta.iter_rows(named=True)}
        self._item_categories = {r["item_id"]: r.get("category", "") for r in item_meta.iter_rows(named=True)}

        # 构建 category / brand → idx 映射
        if "category" in item_meta.columns:
            categories = sorted(item_meta["category"].drop_nulls().unique().to_list())
            self._cat_map = {c: i + 1 for i, c in enumerate(categories)}
            self._cat_map[""] = 0
        else:
            self._cat_map = {}
        if "brand" in item_meta.columns:
            brands = sorted(item_meta["brand"].drop_nulls().unique().to_list())
            self._brand_map = {b: i + 1 for i, b in enumerate(brands)}
            self._brand_map[""] = 0
        else:
            self._brand_map = {}

        self.device = pick_device(device)

        if model_type == "deepfm":
            self.rank_model = self.rank_model.to(self.device).eval()

    def _build_features(self, user_id: str, candidates: list[str]) -> pl.DataFrame:
        """为 (user, candidate_items) 对构建 CTR 特征"""
        u_stat = self._user_stats.get(user_id, {})
        rows = []
        for item_id in candidates:
            i_stat = self._item_stats.get(item_id, {})
            meta = self._item_meta.get(item_id, {})
            rows.append({
                "user_avg_rating": u_stat.get("user_avg_rating", 4.0),
                "user_frequency": float(u_stat.get("user_frequency", 5)),
                "user_active_days": u_stat.get("user_active_days", 0.0),
                "item_avg_rating": i_stat.get("item_avg_rating", 4.0),
                "item_review_count": float(i_stat.get("item_review_count", 1)),
                "item_price_quantile": meta.get("item_price_quantile", 0.5),
                "user_idx": self._uid_map.get(user_id, 0),
                "item_idx": self._iid_map.get(item_id, 0),
                "category_idx": self._cat_map.get(str(meta.get("category", "")), 0) if self._cat_map else 0,
                "brand_idx": self._brand_map.get(str(meta.get("brand", "")), 0) if self._brand_map else 0,
                "weekday": 0,
                "hour": 12,
                "label": 0,
            })
        return pl.DataFrame(rows)

    @torch.no_grad()
    def _rank(self, user_id: str, candidates: list[str]) -> dict[str, float]:
        """对候选集打分，返回 {item_id: score}"""
        feat_df = self._build_features(user_id, candidates)
        if self.model_type == "deepfm":
            from ecom_rec.rank.trainer import prepare_tensors
            dense_t, sparse_t, _ = prepare_tensors(feat_df, self.dense_features, self.sparse_features)
            dense_t, sparse_t = dense_t.to(self.device), sparse_t.to(self.device)
            logits = self.rank_model(dense_t, sparse_t).squeeze(-1).cpu().numpy()
            scores_arr = 1 / (1 + np.exp(-logits))  # sigmoid
        else:
            scores_arr = self.rank_model.predict(feat_df)
        return {item_id: float(s) for item_id, s in zip(candidates, scores_arr)}

    def recommend(self, user_id: str) -> dict[str, Any]:
        """
        完整推荐流程。

        Returns:
            {
                "user_id": str,
                "recall_candidates": list[str],   # 200 候选
                "ranked_top50": list[str],         # 精排 Top-50
                "final_top10": list[str],          # MMR 打散 Top-10
                "scores": dict[str, float],        # 精排分数
            }
        """
        # Step 1: 多路召回
        candidates = self.multi_recall.recommend(user_id, k=self.recall_k)
        if not candidates:
            log.warning(f"用户 {user_id} 多路召回返回空列表，使用空结果")
            return {"user_id": user_id, "recall_candidates": [], "ranked_top50": [], "final_top10": [], "scores": {}}

        # Step 2: 精排
        scores = self._rank(user_id, candidates)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        ranked_top50 = [iid for iid, _ in ranked[:self.rank_top_k]]

        # Step 3: MMR 打散
        final_top10 = mmr_rerank(
            candidates=ranked_top50,
            scores=scores,
            item_categories=self._item_categories,
            k=self.final_k,
            lambda_=self.mmr_lambda,
        )

        return {
            "user_id": user_id,
            "recall_candidates": candidates,
            "ranked_top50": ranked_top50,
            "final_top10": final_top10,
            "scores": {k: scores[k] for k in ranked_top50},
        }
