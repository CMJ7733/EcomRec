"""DeepFM: 深度分解机（Factorization Machine + DNN）CTR 预估模型

参考：Guo et al., "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction", IJCAI 2017
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DeepFM(nn.Module):
    """
    DeepFM 模型，由 3 部分组成：
    1. FM 一阶项（线性）
    2. FM 二阶项（特征交叉）：∑ <vi, vj> xi xj
    3. DNN（高阶特征交互）
    最终输出 = sigmoid(FM_1阶 + FM_2阶 + DNN)
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_vocab_sizes: dict[str, int],
        sparse_features: list[str],
        embedding_dim: int = 16,
        dnn_hidden_units: list[int] = (256, 128, 64),
        dropout: float = 0.3,
        l2_reg: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dense_dim = dense_dim
        self.sparse_features = sparse_features
        self.embedding_dim = embedding_dim
        self.l2_reg = l2_reg

        # Embedding 层（每个稀疏特征独立 Embedding）
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            for feat, vocab_size in sparse_vocab_sizes.items()
            if feat in sparse_features
        })

        # FM 一阶线性部分（稠密特征直接线性变换，稀疏特征 Embedding 后线性）
        n_sparse = len([f for f in sparse_features if f in sparse_vocab_sizes])
        self.fm_linear_dense = nn.Linear(dense_dim, 1, bias=False)
        self.fm_linear_sparse = nn.ModuleDict({
            feat: nn.Linear(embedding_dim, 1, bias=False)
            for feat in sparse_features if feat in sparse_vocab_sizes
        })
        self.fm_bias = nn.Parameter(torch.zeros(1))

        # DNN 输入维度 = dense_dim + n_sparse × embedding_dim
        dnn_input_dim = dense_dim + n_sparse * embedding_dim
        layers: list[nn.Module] = []
        in_dim = dnn_input_dim
        for h in dnn_hidden_units:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.dnn = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, std=0.01)
            nn.init.zeros_(emb.weight[0])  # padding_idx=0 置零
        nn.init.xavier_uniform_(self.fm_linear_dense.weight)
        for lin in self.fm_linear_sparse.values():
            nn.init.xavier_uniform_(lin.weight)
        for module in self.dnn.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dense: (batch, dense_dim) float32
            sparse: (batch, n_sparse) long，列顺序与 self.sparse_features 一致
        Returns:
            (batch, 1) logits（未经 sigmoid）
        """
        # 获取所有稀疏特征的 Embedding
        emb_list = []
        for i, feat in enumerate(self.sparse_features):
            if feat in self.embeddings:
                emb_list.append(self.embeddings[feat](sparse[:, i]))  # (batch, emb_dim)

        # FM 一阶项
        fm1_dense = self.fm_linear_dense(dense)  # (batch, 1)
        fm1_sparse = sum(self.fm_linear_sparse[feat](emb_list[j])
                         for j, feat in enumerate(self.sparse_features)
                         if feat in self.fm_linear_sparse)
        fm1 = fm1_dense + fm1_sparse + self.fm_bias  # (batch, 1)

        # FM 二阶项：∑_{i<j} <vi, vj> = 0.5 * (||∑vi||^2 - ∑||vi||^2)
        if emb_list:
            emb_stack = torch.stack(emb_list, dim=1)  # (batch, n_sparse, emb_dim)
            sum_of_emb = emb_stack.sum(dim=1)         # (batch, emb_dim)
            sum_of_sq = (emb_stack ** 2).sum(dim=1)   # (batch, emb_dim)
            fm2 = 0.5 * ((sum_of_emb ** 2) - sum_of_sq).sum(dim=-1, keepdim=True)  # (batch, 1)
        else:
            fm2 = torch.zeros(dense.size(0), 1, device=dense.device)

        # DNN 部分
        dnn_input = torch.cat([dense] + emb_list, dim=-1)  # (batch, dnn_input_dim)
        dnn_out = self.dnn(dnn_input)  # (batch, 1)

        return fm1 + fm2 + dnn_out  # logits, (batch, 1)

    def l2_loss(self) -> torch.Tensor:
        """L2 正则化损失（用于 Embedding 权重）"""
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for emb in self.embeddings.values():
            reg += (emb.weight ** 2).sum()
        return self.l2_reg * reg
