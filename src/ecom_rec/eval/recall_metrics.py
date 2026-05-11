"""召回评估指标：HR@K, Recall@K, NDCG@K, Coverage, Diversity"""
from __future__ import annotations

import numpy as np
from collections import defaultdict

from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)


def hit_rate_at_k(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    k: int,
) -> float:
    """HR@K：有命中的用户比例（至少有 1 个推荐命中真实交互）"""
    hits = 0
    total = 0
    for user_id, truth in ground_truth.items():
        if user_id not in recommendations:
            continue
        recs = recommendations[user_id][:k]
        truth_set = set(truth)
        if any(r in truth_set for r in recs):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0


def recall_at_k(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    k: int,
) -> float:
    """Recall@K：所有用户的平均召回率"""
    recalls = []
    for user_id, truth in ground_truth.items():
        if user_id not in recommendations:
            continue
        recs = set(recommendations[user_id][:k])
        truth_set = set(truth)
        if len(truth_set) == 0:
            continue
        recalls.append(len(recs & truth_set) / len(truth_set))
    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    k: int,
) -> float:
    """NDCG@K：归一化折损累积增益（考虑命中位置）"""
    ndcgs = []
    for user_id, truth in ground_truth.items():
        if user_id not in recommendations:
            continue
        recs = recommendations[user_id][:k]
        truth_set = set(truth)
        # DCG
        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, item in enumerate(recs)
            if item in truth_set
        )
        # IDCG（理想情况下命中数 = min(|truth|, k)）
        ideal_hits = min(len(truth_set), k)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def coverage(
    recommendations: dict[str, list[str]],
    all_items: set[str],
    k: int,
) -> float:
    """目录覆盖率：被推荐过的商品占全量商品的比例"""
    recommended = set()
    for recs in recommendations.values():
        recommended.update(recs[:k])
    return len(recommended) / len(all_items) if all_items else 0.0


# Alias for backward compatibility
def coverage_at_k(
    recommendations: dict[str, list[str]],
    all_items: set[str],
    k: int,
) -> float:
    """Coverage@K（coverage 的别名，保持向后兼容）"""
    return coverage(recommendations, all_items, k)


def evaluate_recall(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    all_items: set[str],
    k_list: list[int] = (10, 50, 100),
) -> dict[str, float]:
    """
    批量计算多个 K 值下的所有召回指标。

    Returns:
        dict，key 格式如 "HR@10", "Recall@50", "NDCG@10", "Coverage@50"
    """
    results = {}
    for k in k_list:
        results[f"HR@{k}"] = hit_rate_at_k(recommendations, ground_truth, k)
        results[f"Recall@{k}"] = recall_at_k(recommendations, ground_truth, k)
        results[f"NDCG@{k}"] = ndcg_at_k(recommendations, ground_truth, k)
    results[f"Coverage@{max(k_list)}"] = coverage(recommendations, all_items, max(k_list))
    return results
