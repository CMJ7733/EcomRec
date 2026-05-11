"""RFM 指标计算测试"""
import sys
sys.path.insert(0, "src")

import polars as pl
import pytest
from ecom_rec.features.rfm import compute_rfm
from ecom_rec.features.profile import cluster_users, find_optimal_k


def make_test_df() -> pl.DataFrame:
    return pl.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2", "u3"],
        "item_id": ["i1", "i2", "i3", "i1", "i4", "i5"],
        "rating": [5.0, 4.0, 3.0, 5.0, 4.0, 2.0],
        "timestamp_sec": [1000000, 1100000, 1200000, 900000, 1150000, 500000],
        "category": ["skincare", "skincare", "makeup", "makeup", "skincare", "makeup"],
        "price": [20.0, 15.0, None, 30.0, 25.0, 10.0],
    })


def test_rfm_user_count():
    """每个用户应有且只有一条 RFM 记录"""
    df = make_test_df()
    rfm = compute_rfm(df, reference_ts=1300000)
    assert rfm["user_id"].n_unique() == 3
    assert len(rfm) == 3


def test_rfm_frequency():
    """Frequency 应等于该用户的交互次数"""
    df = make_test_df()
    rfm = compute_rfm(df, reference_ts=1300000)
    u1_freq = rfm.filter(pl.col("user_id") == "u1")["frequency"][0]
    assert u1_freq == 3
    u3_freq = rfm.filter(pl.col("user_id") == "u3")["frequency"][0]
    assert u3_freq == 1


def test_rfm_recency_ordering():
    """Recency 最小的用户应是最后一次交互时间最近的"""
    df = make_test_df()
    rfm = compute_rfm(df, reference_ts=1300000)
    # u1 last_ts=1200000, u2 last_ts=1150000, u3 last_ts=500000
    # 所以 u1 应有最小的 recency_days
    min_recency_user = rfm.sort("recency_days")["user_id"][0]
    assert min_recency_user == "u1"


def test_rfm_monetary_non_negative():
    """Monetary 值应非负"""
    df = make_test_df()
    rfm = compute_rfm(df, reference_ts=1300000)
    assert (rfm["monetary"] >= 0).all()


def test_cluster_users_k4():
    """聚类后应有 k 个不同的 cluster_id"""
    df = make_test_df()
    # 需要至少 k 个用户，这里只有 3 个，用 k=2 测试
    rfm = compute_rfm(df, reference_ts=1300000)
    result = cluster_users(rfm, n_clusters=2, random_state=42)
    assert "cluster_id" in result.columns
    assert "user_segment" in result.columns
    assert result["cluster_id"].n_unique() == 2


def test_find_optimal_k_returns_inertias():
    """find_optimal_k 应返回正确长度的 inertia 列表"""
    df = make_test_df()
    rfm = compute_rfm(df, reference_ts=1300000)
    # 使用 k=2 到 3（用户数只有 3）
    inertias = find_optimal_k(rfm, k_range=range(2, 3))
    assert len(inertias) == 1
    assert inertias[0] >= 0
