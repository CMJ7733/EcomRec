"""K-core 过滤逻辑测试"""
import polars as pl
import pytest

import sys
sys.path.insert(0, "src")

from ecom_rec.data.clean import _kcore_filter


def make_interactions(rows: list[tuple[str, str]]) -> pl.DataFrame:
    users, items = zip(*rows)
    return pl.DataFrame({
        "user_id": list(users),
        "item_id": list(items),
        "rating": [5.0] * len(rows),
        "timestamp": [1000] * len(rows),
    })


def test_kcore_basic():
    """K=2 过滤后，所有保留的用户/商品交互数均 >= 2"""
    df = make_interactions([
        ("u1", "i1"), ("u1", "i2"), ("u1", "i3"),
        ("u2", "i1"), ("u2", "i2"),
        ("u3", "i1"),  # u3 只有1次，应被过滤
        ("u4", "i99"),  # i99 只有1次，u4 也应被过滤
    ])
    result = _kcore_filter(df, k=2)
    for uid in result["user_id"].to_list():
        cnt = result.filter(pl.col("user_id") == uid).shape[0]
        assert cnt >= 2, f"用户 {uid} 过滤后仍只有 {cnt} 次交互"
    for iid in result["item_id"].to_list():
        cnt = result.filter(pl.col("item_id") == iid).shape[0]
        assert cnt >= 2, f"商品 {iid} 过滤后仍只有 {cnt} 次交互"


def test_kcore_removes_cascading():
    """K-core 应迭代过滤：当一个商品被移除后，相关用户可能也需要移除"""
    df = make_interactions([
        ("u1", "i1"), ("u1", "i2"),
        ("u2", "i1"), ("u2", "i2"),
        ("u3", "i2"), ("u3", "i3"),  # u3-i3 唯一，导致 i3 被移除，然后 u3 可能交互数不足
    ])
    result = _kcore_filter(df, k=2)
    # i3 应被移除（只有 u3 与之交互）
    assert "i3" not in result["item_id"].to_list()


def test_kcore_all_survive():
    """所有用户/商品均满足 K=2 时，不应移除任何数据"""
    df = make_interactions([
        ("u1", "i1"), ("u1", "i2"),
        ("u2", "i1"), ("u2", "i2"),
    ])
    result = _kcore_filter(df, k=2)
    assert len(result) == 4


def test_kcore_empty_result():
    """当数据不满足 K=3 时，结果应为空"""
    df = make_interactions([
        ("u1", "i1"), ("u1", "i2"),
        ("u2", "i1"),
    ])
    result = _kcore_filter(df, k=3)
    assert len(result) == 0
