"""特征工程层：RFM 建模、CTR 特征"""
from .rfm import compute_rfm
from .profile import cluster_users, find_optimal_k
from .ctr_features import build_ctr_features, save_ctr_features, FeatureSpec

__all__ = ["compute_rfm", "cluster_users", "find_optimal_k", "build_ctr_features", "save_ctr_features", "FeatureSpec"]
