"""验证三个排序模型的接口正确性（不需要真实数据，用构造数据）"""
import sys
sys.path.insert(0, "src")
import torch
import polars as pl
import numpy as np
from ecom_rec.rank.deepfm import DeepFM
from ecom_rec.rank.widedeep import WideDeep
from ecom_rec.rank.lgb import LGBRanker
from ecom_rec.rank.trainer import prepare_tensors, train_model

# 构造小型测试数据
DENSE_FEATURES = ["user_avg_rating", "user_frequency", "user_active_days",
                   "item_avg_rating", "item_review_count", "item_price_quantile"]
SPARSE_FEATURES = ["user_idx", "item_idx", "category_idx", "brand_idx", "weekday", "hour"]
VOCAB_SIZES = {"user_idx": 10, "item_idx": 20, "category_idx": 5, "brand_idx": 8, "weekday": 7, "hour": 24}

n = 100
rng = np.random.default_rng(42)
data = {f: rng.random(n).tolist() for f in DENSE_FEATURES}
data.update({f: rng.integers(1, VOCAB_SIZES[f], n).tolist() for f in SPARSE_FEATURES})
data["label"] = rng.integers(0, 2, n).tolist()
df = pl.DataFrame(data)
train_df = df.head(80)
valid_df = df.tail(20)

# 1. 验证 DeepFM 前向传播
model = DeepFM(
    dense_dim=len(DENSE_FEATURES),
    sparse_vocab_sizes=VOCAB_SIZES,
    sparse_features=SPARSE_FEATURES,
    embedding_dim=8,
    dnn_hidden_units=[32, 16],
)
dense_t, sparse_t, labels_t = prepare_tensors(train_df.head(4), DENSE_FEATURES, SPARSE_FEATURES)
out = model(dense_t, sparse_t)
assert out.shape == (4, 1), f"DeepFM 输出形状错误：{out.shape}"
print(f"✓ DeepFM 前向传播：输出形状 {out.shape}")

# 2. 验证 Wide&Deep 前向传播
wd = WideDeep(
    dense_dim=len(DENSE_FEATURES),
    sparse_vocab_sizes=VOCAB_SIZES,
    sparse_features=SPARSE_FEATURES,
    embedding_dim=8,
    dnn_hidden_units=[32, 16],
)
out_wd = wd(dense_t, sparse_t)
assert out_wd.shape == (4, 1), f"WideDeep 输出形状错误：{out_wd.shape}"
print(f"✓ Wide&Deep 前向传播：输出形状 {out_wd.shape}")

# 3. 验证 DeepFM 小规模训练（2 epoch）
small_model = DeepFM(
    dense_dim=len(DENSE_FEATURES),
    sparse_vocab_sizes=VOCAB_SIZES,
    sparse_features=SPARSE_FEATURES,
    embedding_dim=8,
    dnn_hidden_units=[32, 16],
)
history = train_model(
    small_model, train_df, valid_df,
    dense_features=DENSE_FEATURES,
    sparse_features=SPARSE_FEATURES,
    epochs=2, batch_size=32, lr=1e-3, patience=5, use_amp=False,
)
assert len(history["val_auc"]) == 2
assert all(0.0 <= a <= 1.0 for a in history["val_auc"])
print(f"✓ DeepFM 训练循环：2 epoch，最终 val_AUC={history['val_auc'][-1]:.4f}")

# 4. 验证 LightGBM
lgb_model = LGBRanker(n_estimators=10, early_stopping_rounds=5)
lgb_model.fit(train_df, valid_df, DENSE_FEATURES, SPARSE_FEATURES)
preds = lgb_model.predict(valid_df)
assert len(preds) == 20
assert all(0.0 <= p <= 1.0 for p in preds)
print(f"✓ LightGBM 训练完成，预测形状：{preds.shape}")

fi = lgb_model.feature_importance()
assert len(fi) == len(DENSE_FEATURES) + len(SPARSE_FEATURES)
print(f"✓ 特征重要度：{fi['feature'].to_list()[:3]} 最重要")

print("\n所有排序模型验证通过 ✓")
