"""排序评估指标：AUC, LogLoss, GAUC"""
from __future__ import annotations

import numpy as np
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
    关注用户内部排序质量，比全局 AUC 更贴近真实推荐场景。
    """
    gauc_sum = 0.0
    total_weight = 0
    for uid in np.unique(user_ids):
        mask = user_ids == uid
        y_t = y_true[mask]
        y_s = y_score[mask]
        if len(np.unique(y_t)) < 2:
            continue  # 该用户全正或全负，跳过
        try:
            auc = roc_auc_score(y_t, y_s)
            weight = mask.sum()
            gauc_sum += auc * weight
            total_weight += weight
        except ValueError:
            continue
    return gauc_sum / total_weight if total_weight > 0 else 0.0
