"""通用深度模型训练循环：支持 AMP 混合精度、EarlyStopping、LR Scheduler"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from ecom_rec.eval.rank_metrics import compute_auc, compute_logloss
from ecom_rec.utils.device import pick_device
from ecom_rec.utils.logger import get_logger
from ecom_rec.utils.seed import set_seed

log = get_logger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, score: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            import copy
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def prepare_tensors(
    df,  # polars DataFrame
    dense_features: list[str],
    sparse_features: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 polars DataFrame 转为 (dense_tensor, sparse_tensor, label_tensor)"""
    import polars as pl
    dense_arr = df.select(dense_features).to_numpy()
    dense_arr = np.nan_to_num(dense_arr, nan=0.0)
    sparse_arr = df.select(sparse_features).to_numpy()
    sparse_arr = np.nan_to_num(sparse_arr, nan=0).astype(np.int64)
    dense = torch.tensor(dense_arr, dtype=torch.float32)
    sparse = torch.tensor(sparse_arr, dtype=torch.long)
    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)
    return dense, sparse, labels


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None,
    desc: str = "train",
) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for dense, sparse, labels in pbar:
        dense, sparse, labels = dense.to(device), sparse.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with autocast(device_type="cuda"):
                preds = model(dense, sparse).squeeze(-1)
                loss = criterion(preds, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(dense, sparse).squeeze(-1)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """返回 (AUC, LogLoss)"""
    model.eval()
    all_preds, all_labels = [], []
    for dense, sparse, labels in tqdm(loader, desc="eval", leave=False, unit="batch"):
        dense, sparse = dense.to(device), sparse.to(device)
        preds = torch.sigmoid(model(dense, sparse)).squeeze(-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    return compute_auc(labels_arr, preds_arr), compute_logloss(labels_arr, preds_arr)


def train_model(
    model: nn.Module,
    train_df,
    valid_df,
    dense_features: list[str],
    sparse_features: list[str],
    epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 1e-3,
    patience: int = 3,
    use_amp: bool = True,
    save_path: str | None = None,
    random_state: int = 42,
    sample_ratio: float = 1.0,
) -> dict[str, list[float]]:
    """
    训练深度排序模型。

    Args:
        sample_ratio: 训练集下采样比例，如 0.1 表示只用 10% 数据训练，加速调试。

    Returns:
        history dict，含 train_loss, val_auc, val_logloss 列表
    """
    set_seed(random_state)
    device = pick_device()
    log.info(f"训练设备：{device}")

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
    scaler = GradScaler(device="cuda") if (use_amp and device.type == "cuda") else None
    early_stop = EarlyStopping(patience=patience)

    # 训练集下采样
    if sample_ratio < 1.0:
        import polars as pl
        n = len(train_df)
        n_sample = max(int(n * sample_ratio), 1)
        train_df = train_df.sample(n=n_sample, seed=random_state)
        log.info(f"训练集下采样：{n:,} → {len(train_df):,}（比例={sample_ratio}）")

    # 构建 DataLoader
    train_dense, train_sparse, train_labels = prepare_tensors(train_df, dense_features, sparse_features)
    val_dense, val_sparse, val_labels = prepare_tensors(valid_df, dense_features, sparse_features)
    pin = device.type == "cuda"
    train_loader = DataLoader(TensorDataset(train_dense, train_sparse, train_labels),
                              batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin)
    val_loader = DataLoader(TensorDataset(val_dense, val_sparse, val_labels),
                            batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=pin)

    history = {"train_loss": [], "val_auc": [], "val_logloss": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler,
                                  desc=f"epoch {epoch}/{epochs}")
        val_auc, val_logloss = evaluate(model, val_loader, device)
        scheduler.step(val_logloss)

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        history["val_logloss"].append(val_logloss)

        log.info(f"Epoch {epoch:02d}/{epochs} | loss={train_loss:.4f} | val_auc={val_auc:.4f} | val_logloss={val_logloss:.4f}")

        if early_stop.step(val_auc, model):
            log.info(f"Early stopping 触发（patience={patience}），最优 AUC={early_stop.best_score:.4f}")
            early_stop.restore_best(model)
            break

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        log.info(f"模型已保存到 {save_path}")

    return history
