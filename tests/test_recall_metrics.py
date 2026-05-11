"""召回指标计算正确性测试"""
import sys
sys.path.insert(0, "src")

import pytest
from ecom_rec.eval.recall_metrics import (
    hit_rate_at_k,
    recall_at_k,
    ndcg_at_k,
    coverage,
    evaluate_recall,
)


# 构造确定性测试数据
RECS = {
    "u1": ["i1", "i2", "i3", "i4", "i5"],  # i1 命中
    "u2": ["i6", "i7", "i8", "i9", "i10"],  # 无命中
    "u3": ["i11", "i1", "i12"],              # i1 在第 2 位命中
}
TRUTH = {
    "u1": ["i1"],
    "u2": ["i99"],  # i99 不在推荐列表
    "u3": ["i1"],
}
ALL_ITEMS = {f"i{j}" for j in range(1, 101)}


def test_hr_k_perfect():
    """完全命中时 HR@K = 1.0"""
    recs = {"u1": ["i1", "i2"]}
    truth = {"u1": ["i1"]}
    assert hit_rate_at_k(recs, truth, k=5) == 1.0


def test_hr_k_no_hit():
    """完全未命中时 HR@K = 0.0"""
    recs = {"u1": ["i2", "i3"]}
    truth = {"u1": ["i1"]}
    assert hit_rate_at_k(recs, truth, k=5) == 0.0


def test_hr_k_partial():
    """部分命中：2/3 用户命中"""
    result = hit_rate_at_k(RECS, TRUTH, k=5)
    assert abs(result - 2 / 3) < 1e-9


def test_recall_at_k_single_truth():
    """单个真实样本，命中时 Recall = 1.0"""
    recs = {"u1": ["i1", "i2"]}
    truth = {"u1": ["i1"]}
    assert recall_at_k(recs, truth, k=5) == 1.0


def test_recall_at_k_no_hit():
    """无命中时 Recall = 0.0"""
    recs = {"u1": ["i2", "i3"]}
    truth = {"u1": ["i1"]}
    assert recall_at_k(recs, truth, k=5) == 0.0


def test_ndcg_position_matters():
    """NDCG 应该对命中位置敏感：越早命中 NDCG 越高"""
    recs_early = {"u1": ["i1", "i2", "i3"]}  # i1 在第 1 位
    recs_late = {"u1": ["i2", "i3", "i1"]}   # i1 在第 3 位
    truth = {"u1": ["i1"]}
    ndcg_early = ndcg_at_k(recs_early, truth, k=5)
    ndcg_late = ndcg_at_k(recs_late, truth, k=5)
    assert ndcg_early > ndcg_late


def test_ndcg_perfect():
    """完美推荐时 NDCG = 1.0"""
    recs = {"u1": ["i1"]}
    truth = {"u1": ["i1"]}
    assert abs(ndcg_at_k(recs, truth, k=5) - 1.0) < 1e-9


def test_coverage_full():
    """所有商品都被推荐时 Coverage = 1.0"""
    all_items = {"i1", "i2", "i3"}
    recs = {"u1": ["i1", "i2"], "u2": ["i3"]}
    assert coverage(recs, all_items, k=5) == 1.0


def test_coverage_partial():
    """只有部分商品被推荐"""
    all_items = {"i1", "i2", "i3", "i4"}
    recs = {"u1": ["i1", "i2"]}
    assert coverage(recs, all_items, k=5) == 0.5


def test_evaluate_recall_keys():
    """evaluate_recall 应返回包含所有指标 key 的 dict"""
    result = evaluate_recall(RECS, TRUTH, ALL_ITEMS, k_list=[10, 50])
    assert "HR@10" in result
    assert "Recall@10" in result
    assert "NDCG@10" in result
    assert "HR@50" in result
    assert "Coverage@50" in result
