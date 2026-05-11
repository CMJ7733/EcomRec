"""验证召回层接口正确性的脚本（纯 CPU，无需 GPU）"""
import sys
import os

# 将 src 目录添加到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import polars as pl
from ecom_rec.recall.pop import PopRecaller
from ecom_rec.recall.itemcf import ItemCFRecaller

df = pl.DataFrame({
    "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u3", "u4"],
    "item_id": ["i1", "i2", "i3", "i1", "i4", "i2", "i3", "i5"],
    "rating": [5, 4, 3, 5, 4, 3, 5, 4],
    "timestamp_sec": [100, 200, 300, 150, 250, 180, 280, 100],
})

# --- PopRecaller 验证 ---
print("=== 测试 PopRecaller ===")
pop = PopRecaller().fit(df)
recs = pop.recommend("u1", k=5)
print(f"  u1 的推荐结果：{recs}")
assert len(recs) <= 5, f"结果数量超过 k=5：{len(recs)}"
assert "i1" not in recs, "i1 应被过滤（u1 已交互）"
assert "i2" not in recs, "i2 应被过滤（u1 已交互）"
assert "i3" not in recs, "i3 应被过滤（u1 已交互）"
print("  PopRecaller 验证通过")

# --- 批量推荐验证 ---
batch_recs = pop.recommend_batch(["u1", "u2", "u4"], k=3)
assert isinstance(batch_recs, dict), "recommend_batch 应返回 dict"
assert "u1" in batch_recs and "u2" in batch_recs and "u4" in batch_recs
print(f"  recommend_batch 结果：{batch_recs}")
print("  recommend_batch 验证通过")

# --- ItemCFRecaller 验证 ---
print("=== 测试 ItemCFRecaller ===")
icf = ItemCFRecaller(n_neighbors=5).fit(df)
recs_icf = icf.recommend("u1", k=5)
print(f"  u1 的 ItemCF 推荐结果：{recs_icf}")
assert isinstance(recs_icf, list), "recommend 应返回 list"
assert len(recs_icf) <= 5, f"结果数量超过 k=5：{len(recs_icf)}"
# u1 历史商品不应出现在推荐结果中
u1_history = {"i1", "i2", "i3"}
for item in recs_icf:
    assert item not in u1_history, f"历史商品 {item} 不应出现在推荐结果中"
print("  ItemCFRecaller 验证通过")

# --- 冷启动验证（未知用户） ---
print("=== 测试冷启动（未知用户） ===")
recs_unknown = pop.recommend("unknown_user", k=5)
assert isinstance(recs_unknown, list)
print(f"  未知用户 PopRecaller 推荐：{recs_unknown}")

recs_icf_unknown = icf.recommend("unknown_user", k=5)
assert isinstance(recs_icf_unknown, list)
print(f"  未知用户 ItemCF 推荐：{recs_icf_unknown}（预期为空列表）")

# --- name 属性验证 ---
assert pop.name == "PopRecaller"
assert icf.name == "ItemCFRecaller"
print("  name 属性验证通过")

print("\n召回接口验证通过")
