"""验证 CTR 特征构建逻辑"""
import sys
sys.path.insert(0, "src")
import polars as pl
from ecom_rec.features.ctr_features import build_ctr_features

# 构造小数据集
train = pl.DataFrame({
    "user_id": ["u1","u1","u2","u2","u3"],
    "item_id": ["i1","i2","i1","i3","i2"],
    "rating": [5.0,4.0,3.0,5.0,4.0],
    "timestamp_sec": [100,200,150,300,250],
    "weekday": [1,2,3,4,5],
    "hour": [10,11,12,13,14],
    "category": ["skincare","makeup","skincare","makeup","skincare"],
    "brand": ["brandA","brandB","brandA","brandC","brandB"],
    "price": [20.0, 15.0, 20.0, 30.0, 15.0],
})
valid = train.head(2)
test = train.tail(2)
user_map = pl.DataFrame({"user_id": ["u1","u2","u3"], "user_idx": [0,1,2]})
item_map = pl.DataFrame({"item_id": ["i1","i2","i3"], "item_idx": [0,1,2]})

train_f, valid_f, test_f, spec = build_ctr_features(
    train, valid, test, user_map, item_map, neg_sample_ratio=2, random_state=42
)

# 验证
assert "label" in train_f.columns
assert "user_idx" in train_f.columns
assert "item_price_quantile" in train_f.columns
assert train_f["label"].sum() == len(train)  # 正样本数 = 原始 train 大小
assert len(train_f) == len(train) * (1 + 2)  # 正 + 2x 负
neg_ratio = (train_f["label"] == 0).sum() / (train_f["label"] == 1).sum()
assert abs(neg_ratio - 2.0) < 0.1
assert spec.sparse_vocab_sizes["weekday"] == 7
assert spec.sparse_vocab_sizes["hour"] == 24
print(f"训练集：{len(train_f)} 条（正:{train_f['label'].sum()} 负:{len(train_f)-train_f['label'].sum()}）")
print(f"特征列：{train_f.columns}")
print(f"特征规格：{spec}")
print("CTR 特征验证全部通过 [PASS]")
