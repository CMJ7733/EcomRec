"""Wide & Deep Learning for Recommender Systems

参考：Cheng et al., "Wide & Deep Learning for Recommender Systems", RecSys 2016
Wide 侧：线性模型（LR），处理记忆效应
Deep 侧：DNN，处理泛化效应
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WideDeep(nn.Module):
    """Wide & Deep 推荐模型。"""

    def __init__(
        self,
        dense_dim: int,
        sparse_vocab_sizes: dict[str, int],
        sparse_features: list[str],
        embedding_dim: int = 16,
        dnn_hidden_units: list[int] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.sparse_features = sparse_features

        # Embedding 层（Deep 侧使用）
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            for feat, vocab_size in sparse_vocab_sizes.items()
            if feat in sparse_features
        })

        # Wide 侧：稠密特征直接线性映射
        self.wide = nn.Linear(dense_dim, 1)

        # Deep 侧：DNN
        n_sparse = len([f for f in sparse_features if f in sparse_vocab_sizes])
        dnn_input_dim = dense_dim + n_sparse * embedding_dim
        layers: list[nn.Module] = []
        in_dim = dnn_input_dim
        for h in dnn_hidden_units:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.deep = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, std=0.01)
        nn.init.zeros_(self.wide.bias)
        for module in self.deep.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        emb_list = [
            self.embeddings[feat](sparse[:, i])
            for i, feat in enumerate(self.sparse_features)
            if feat in self.embeddings
        ]
        wide_out = self.wide(dense)  # (batch, 1)
        dnn_input = torch.cat([dense] + emb_list, dim=-1)
        deep_out = self.deep(dnn_input)  # (batch, 1)
        return wide_out + deep_out  # logits
