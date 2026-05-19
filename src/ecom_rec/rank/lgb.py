"""LightGBM CTR 精排模型（Pointwise 二分类）"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb

from ecom_rec.eval.rank_metrics import compute_auc, compute_logloss, compute_gauc
from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)


class LGBRanker:
    """LightGBM Pointwise/Pairwise CTR 模型封装。"""

    def __init__(
        self,
        objective: str = "binary",
        num_leaves: int = 127,
        learning_rate: float = 0.05,
        n_estimators: int = 500,
        early_stopping_rounds: int = 20,
        random_state: int = 42,
    ) -> None:
        self.params = {
            "objective": objective,
            "metric": "auc",
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "random_state": random_state,
            "verbose": -1,
        }
        self.early_stopping_rounds = early_stopping_rounds
        self._model: lgb.Booster | None = None
        self._feature_names: list[str] = []

    def fit(
        self,
        train_df: pl.DataFrame,
        valid_df: pl.DataFrame,
        dense_features: list[str],
        sparse_features: list[str],
    ) -> "LGBRanker":
        # 稀疏特征是整数 ID，专供 Embedding 查表，对树模型无意义，只用稠密特征
        feature_cols = dense_features
        self._feature_names = feature_cols

        X_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df["label"].to_numpy()
        X_valid = valid_df.select(feature_cols).to_numpy()
        y_valid = valid_df["label"].to_numpy()

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, feature_name=feature_cols)

        callbacks = [
            lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ]

        self._model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params["n_estimators"],
            valid_sets=[dvalid],
            callbacks=callbacks,
        )

        val_preds = self._model.predict(X_valid)
        val_auc = compute_auc(y_valid, val_preds)
        val_logloss = compute_logloss(y_valid, val_preds)
        log.info(f"LightGBM 训练完成：val_AUC={val_auc:.4f}  val_LogLoss={val_logloss:.4f}")
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        X = df.select(self._feature_names).to_numpy()
        return self._model.predict(X)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(path)
        log.info(f"LightGBM 模型已保存到 {path}")

    @classmethod
    def load(cls, path: str) -> "LGBRanker":
        obj = cls()
        obj._model = lgb.Booster(model_file=path)
        return obj

    def feature_importance(self) -> pl.DataFrame:
        """返回特征重要度 DataFrame"""
        names = self._model.feature_name()
        gains = self._model.feature_importance(importance_type="gain")
        return (
            pl.DataFrame({"feature": names, "importance": gains.tolist()})
            .sort("importance", descending=True)
        )
