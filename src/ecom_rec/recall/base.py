"""召回模型抽象基类"""
from __future__ import annotations
from abc import ABC, abstractmethod
import polars as pl


class Recaller(ABC):
    """所有召回模型的抽象基类。

    接口约定：
    - fit(interactions): 训练/拟合模型
    - recommend(user_id, k): 返回 Top-K 候选商品 ID 列表
    - recommend_batch(user_ids, k): 批量推荐（默认逐一调用 recommend）
    """

    @abstractmethod
    def fit(self, interactions: pl.DataFrame) -> "Recaller":
        """
        Args:
            interactions: 含 user_id, item_id 列的交互 DataFrame
        Returns:
            self（支持链式调用）
        """

    @abstractmethod
    def recommend(self, user_id: str, k: int = 50) -> list[str]:
        """
        Args:
            user_id: 用户 ID
            k: 返回候选数量
        Returns:
            商品 ID 列表（长度 <= k，按相关度降序）
        """

    def recommend_batch(self, user_ids: list[str], k: int = 50) -> dict[str, list[str]]:
        """批量推荐，子类可覆盖以实现并行加速"""
        return {uid: self.recommend(uid, k) for uid in user_ids}

    @property
    def name(self) -> str:
        return self.__class__.__name__
