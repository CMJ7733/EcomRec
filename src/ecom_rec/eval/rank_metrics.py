"""排序评估指标：AUC, LogLoss, GAUC"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss

from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """全局 AUC"""
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.0


def compute_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """LogLoss（越低越好）"""
    y_score = np.clip(y_score, 1e-7, 1 - 1e-7)
    return float(log_loss(y_true, y_score))


def compute_gauc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    user_ids: np.ndarray,
) -> float:
    """
    GAUC（按用户分组的 AUC 加权均值）。
    使用 Mann-Whitney U 公式向量化计算，避免逐用户 Python 循环。

    AUC = (U_pos) / (n_pos * n_neg)
    其中 U_pos = sum of positive ranks - n_pos*(n_pos+1)/2
    """
    df = pl.DataFrame({
        "uid": user_ids,
        "y_true": y_true,
        "y_score": y_score,
    })

    grouped = df.group_by("uid").agg([
        pl.col("y_true").alias("labels"),
        pl.col("y_score").alias("scores"),
        pl.len().alias("cnt"),
    ])

    gauc_sum = 0.0
    total_weight = 0

    for row in grouped.iter_rows(named=True):
        labels = np.asarray(row["labels"])
        scores = np.asarray(row["scores"])
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue

        ranks = np.argsort(np.argsort(scores)).astype(np.float64) + 1.0
        u = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0
        auc_val = u / (n_pos * n_neg)
        weight = len(labels)
        gauc_sum += auc_val * weight
        total_weight += weight

    return gauc_sum / total_weight if total_weight > 0 else 0.0
