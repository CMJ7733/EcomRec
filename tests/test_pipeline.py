"""推荐流水线测试：MultiRecall 融合 + MMR 打散"""
import sys
sys.path.insert(0, "src")

import polars as pl
import pytest

from ecom_rec.pipeline.multi_recall import MultiRecall
from ecom_rec.pipeline.rerank import mmr_rerank
from ecom_rec.recall.base import Recaller


class _MockRecaller(Recaller):
    """Mock：构造时直接给定 user → list[item] 映射，无需 fit"""

    def __init__(self, mapping: dict[str, list[str]]):
        self._mapping = mapping

    def fit(self, interactions):
        return self

    def recommend(self, user_id: str, k: int = 50) -> list[str]:
        return self._mapping.get(user_id, [])[:k]


# ---- MultiRecall ----------------------------------------------------------

def test_multi_recall_fusion_top_item():
    """两路召回的第 1 名应该融合后仍排在最前面"""
    r1 = _MockRecaller({"u1": ["a", "b", "c"]})
    r2 = _MockRecaller({"u1": ["a", "d", "e"]})
    mr = MultiRecall([(r1, 1.0), (r2, 1.0)])
    out = mr.recommend("u1", k=5)
    assert out[0] == "a", "两路都把 a 排第 1，融合后必须是首位"
    assert set(out) == {"a", "b", "c", "d", "e"}


def test_multi_recall_weight_dominates():
    """权重高的召回器排第 1 的物品应胜过权重低的"""
    r1 = _MockRecaller({"u1": ["x", "y"]})
    r2 = _MockRecaller({"u1": ["z", "x"]})
    mr_low = MultiRecall([(r1, 0.1), (r2, 0.9)])  # r2 主导
    assert mr_low.recommend("u1", k=2)[0] == "z"

    mr_high = MultiRecall([(r1, 0.9), (r2, 0.1)])  # r1 主导
    assert mr_high.recommend("u1", k=2)[0] == "x"


def test_multi_recall_batch():
    """recommend_batch 应返回 dict 且 key 一一对应"""
    r1 = _MockRecaller({"u1": ["a"], "u2": ["b"]})
    mr = MultiRecall([(r1, 1.0)])
    result = mr.recommend_batch(["u1", "u2", "u3"], k=5)
    assert set(result.keys()) == {"u1", "u2", "u3"}
    assert result["u3"] == []  # 未知用户


# ---- MMR rerank ----------------------------------------------------------

def test_mmr_lambda_1_pure_relevance():
    """λ=1 应严格按 score 降序，不考虑多样性"""
    cands = ["i1", "i2", "i3"]
    scores = {"i1": 0.9, "i2": 0.5, "i3": 0.3}
    cats = {"i1": "A", "i2": "A", "i3": "B"}
    out = mmr_rerank(cands, scores, cats, k=3, lambda_=1.0)
    assert out == ["i1", "i2", "i3"]


def test_mmr_lambda_0_pure_diversity():
    """λ=0：第 2 选起优先选与已选类目不同的候选"""
    cands = ["i1", "i2", "i3"]
    scores = {"i1": 0.9, "i2": 0.8, "i3": 0.3}
    cats = {"i1": "A", "i2": "A", "i3": "B"}
    out = mmr_rerank(cands, scores, cats, k=2, lambda_=0.0)
    assert out[0] == "i1"        # 第一个始终选最高分
    assert out[1] == "i3"        # 第二个选与 i1 类目不同的 i3


def test_mmr_empty_candidates():
    """空候选应返回空列表，不报错"""
    assert mmr_rerank([], {}, {}, k=5, lambda_=0.5) == []


def test_mmr_k_capped_by_candidates():
    """k 超过候选数时只返回所有候选"""
    out = mmr_rerank(["i1", "i2"], {"i1": 1.0, "i2": 0.5}, {"i1": "A", "i2": "B"},
                      k=10, lambda_=0.5)
    assert len(out) == 2
