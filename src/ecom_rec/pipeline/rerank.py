"""
精排后处理：MMR（Maximal Marginal Relevance）打散算法

MMR 在相关性（来自精排分数）和多样性（商品类目相似度）之间取平衡：
score_mmr(i) = λ × rel(i) - (1-λ) × max_{j ∈ selected} sim(i, j)

参考：Carbonell & Goldstein, "The Use of MMR for Diversity-Based Reranking", SIGIR 1998
"""
from __future__ import annotations

import numpy as np


def mmr_rerank(
    candidates: list[str],
    scores: dict[str, float],
    item_categories: dict[str, str],
    k: int = 10,
    lambda_: float = 0.5,
) -> list[str]:
    """
    MMR 打散：在相关性与类目多样性之间取平衡。

    Args:
        candidates: 候选商品 ID 列表（已按精排分数排序）
        scores: 精排分数字典 {item_id: score}
        item_categories: 商品类目字典 {item_id: category}
        k: 最终返回数量
        lambda_: 相关性权重（越大越倾向于相关性，越小越倾向于多样性）

    Returns:
        打散后的 Top-K 商品列表
    """
    if not candidates:
        return []

    # 归一化精排分数到 [0, 1]
    all_scores = np.array([scores.get(i, 0.0) for i in candidates])
    s_min, s_max = all_scores.min(), all_scores.max()
    if s_max > s_min:
        normalized = {i: (scores.get(i, 0.0) - s_min) / (s_max - s_min) for i in candidates}
    else:
        normalized = {i: 1.0 for i in candidates}

    selected: list[str] = []
    remaining = list(candidates)

    while len(selected) < k and remaining:
        if not selected:
            # 第一个直接选最高分
            best = max(remaining, key=lambda i: normalized.get(i, 0.0))
        else:
            # MMR 打散选择
            best = None
            best_score = -float("inf")
            for item in remaining:
                rel = lambda_ * normalized.get(item, 0.0)
                # 类目相似度：与已选商品类目相同 → 相似度 = 1，否则 = 0
                selected_cats = {item_categories.get(s, "") for s in selected}
                item_cat = item_categories.get(item, "")
                max_sim = 1.0 if item_cat and item_cat in selected_cats else 0.0
                diversity = (1 - lambda_) * max_sim
                mmr_score = rel - diversity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = item
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)

    return selected
