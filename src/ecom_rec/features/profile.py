"""用户分群：基于 RFM 指标的 KMeans 聚类"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)

# 用户群体标签映射（根据各群体 RFM 均值语义命名）
CLUSTER_LABELS = {
    "high_value": "核心高价值用户",
    "potential": "潜力用户",
    "dormant": "沉睡用户",
    "churned": "流失用户",
}


def _label_clusters(rfm_with_cluster: pl.DataFrame, n_clusters: int) -> pl.DataFrame:
    """根据各簇 RFM 均值，为每个 cluster_id 赋语义标签"""
    cluster_stats = (
        rfm_with_cluster.group_by("cluster_id")
        .agg([
            pl.col("recency_days").mean().alias("avg_recency"),
            pl.col("frequency").mean().alias("avg_frequency"),
            pl.col("monetary").mean().alias("avg_monetary"),
        ])
        .sort("cluster_id")
    )
    stats_pd = cluster_stats.to_pandas()

    # 综合评分：高 F + 高 M + 低 R = 高分
    stats_pd["score"] = (
        stats_pd["avg_frequency"].rank() +
        stats_pd["avg_monetary"].rank() +
        (stats_pd["avg_recency"].max() - stats_pd["avg_recency"]).rank()
    )
    stats_pd = stats_pd.sort_values("score", ascending=False).reset_index(drop=True)

    label_keys = list(CLUSTER_LABELS.keys())
    label_map = {}
    for i, row in stats_pd.iterrows():
        label_key = label_keys[i] if i < len(label_keys) else f"group_{i}"
        label_map[int(row["cluster_id"])] = CLUSTER_LABELS.get(label_key, label_key)

    rfm_with_cluster = rfm_with_cluster.with_columns(
        pl.col("cluster_id").map_elements(
            lambda cid: label_map.get(cid, f"group_{cid}"), return_dtype=pl.Utf8
        ).alias("user_segment")
    )
    return rfm_with_cluster


def cluster_users(
    rfm: pl.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    对 RFM 指标进行 KMeans 聚类。

    Args:
        rfm: compute_rfm() 的输出，含 user_id/recency_days/frequency/monetary
        n_clusters: 聚类数量，默认 4
        random_state: 随机种子

    Returns:
        原 rfm DataFrame 追加 cluster_id 和 user_segment 列
    """
    features = ["recency_days", "frequency", "monetary"]
    X = rfm.select(features).to_numpy().astype(np.float64)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    rfm = rfm.with_columns(pl.Series("cluster_id", labels.tolist(), dtype=pl.Int32))
    rfm = _label_clusters(rfm, n_clusters)

    # 打印各群体统计
    segment_stats = (
        rfm.group_by("user_segment")
        .agg([
            pl.len().alias("user_count"),
            pl.col("recency_days").mean().round(1).alias("avg_recency_days"),
            pl.col("frequency").mean().round(1).alias("avg_frequency"),
            pl.col("monetary").mean().round(2).alias("avg_monetary"),
        ])
        .sort("avg_monetary", descending=True)
    )
    log.info(f"\n用户分群结果（k={n_clusters}）：\n{segment_stats}")
    return rfm


def find_optimal_k(rfm: pl.DataFrame, k_range: range = range(2, 9)) -> list[float]:
    """使用肘部法则寻找最优 K 值，返回各 K 对应的 inertia"""
    features = ["recency_days", "frequency", "monetary"]
    X = rfm.select(features).to_numpy().astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return inertias
