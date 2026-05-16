"""训练 LightGBM / DeepFM / Wide&Deep 三个 CTR 排序模型，并在 test 上做最终评估。

用法：
    python scripts/03_train_rank.py
    python scripts/03_train_rank.py fast=true
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf

from ecom_rec.eval.rank_metrics import compute_auc, compute_gauc, compute_logloss
from ecom_rec.features.ctr_features import build_ctr_features, save_ctr_features
from ecom_rec.rank.deepfm import DeepFM
from ecom_rec.rank.lgb import LGBRanker
from ecom_rec.rank.trainer import prepare_tensors, train_model
from ecom_rec.rank.widedeep import WideDeep
from ecom_rec.utils.device import pick_device
from ecom_rec.utils.io import read_parquet
from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)


def _load_cfg() -> OmegaConf:
    root = OmegaConf.load("configs/config.yaml")
    rank_cfgs = {
        "lgb": OmegaConf.load("configs/rank/lgb.yaml"),
        "deepfm": OmegaConf.load("configs/rank/deepfm.yaml"),
        "widedeep": OmegaConf.load("configs/rank/widedeep.yaml"),
    }
    cli = OmegaConf.from_cli()
    return OmegaConf.merge(root, {"rank_cfgs": rank_cfgs}, cli)


def _apply_fast_overrides(cfg: OmegaConf) -> None:
    if not cfg.get("fast", False):
        return
    log.info("[fast] 模式开启：缩减 epoch/n_estimators，训练集下采样")
    cfg.rank_cfgs.lgb.n_estimators = 100
    cfg.rank_cfgs.deepfm.epochs = 5
    cfg.rank_cfgs.widedeep.epochs = 5
    if "sample_ratio" not in cfg:
        cfg.sample_ratio = 0.1


def _ensure_ctr_features(seed: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """若 ctr 特征已存在则直接加载，否则现场构建并落盘。"""
    processed = Path("data/processed")
    paths = [processed / n for n in ("ctr_train.parquet", "ctr_valid.parquet", "ctr_test.parquet")]
    spec_path = processed / "feature_spec.json"

    if all(p.exists() for p in paths) and spec_path.exists():
        log.info("检测到 CTR 特征已存在，跳过构建。")
        train_f = read_parquet(paths[0])
        valid_f = read_parquet(paths[1])
        test_f = read_parquet(paths[2])
        spec = json.loads(spec_path.read_text())
        return train_f, valid_f, test_f, spec

    log.info("构建 CTR 特征 ...")
    train = read_parquet(processed / "train.parquet")
    valid = read_parquet(processed / "valid.parquet")
    test = read_parquet(processed / "test.parquet")
    user_map = read_parquet(processed / "user_map.parquet")
    item_map = read_parquet(processed / "item_map.parquet")

    train_f, valid_f, test_f, fspec = build_ctr_features(
        train, valid, test, user_map, item_map,
        neg_sample_ratio=4, random_state=seed,
    )
    save_ctr_features(train_f, valid_f, test_f, fspec, str(processed))
    spec = {
        "dense_features": fspec.dense_features,
        "sparse_features": fspec.sparse_features,
        "sparse_vocab_sizes": fspec.sparse_vocab_sizes,
    }
    return train_f, valid_f, test_f, spec


def _evaluate_on_test(model_name: str, predictor, test_df: pl.DataFrame) -> dict:
    """统一评估：AUC / LogLoss / GAUC（按 user_idx 分组）"""
    labels = test_df["label"].to_numpy()
    preds = predictor(test_df)
    auc = compute_auc(labels, preds)
    logloss = compute_logloss(labels, preds)
    gauc = compute_gauc(labels, preds, test_df["user_idx"].to_numpy())
    log.info(f"{model_name} test: AUC={auc:.4f}  LogLoss={logloss:.4f}  GAUC={gauc:.4f}")
    return {"AUC": auc, "LogLoss": logloss, "GAUC": gauc}


def _predict_torch(model: torch.nn.Module, device: torch.device,
                   dense_features: list[str], sparse_features: list[str],
                   batch_size: int = 8192):
    """返回一个闭包，接收 DataFrame → 分批推理输出 sigmoid 后概率数组"""
    model.eval()

    @torch.no_grad()
    def _predictor(df: pl.DataFrame) -> np.ndarray:
        dense_t, sparse_t, _ = prepare_tensors(df, dense_features, sparse_features)
        dense_t, sparse_t = dense_t.to(device), sparse_t.to(device)
        all_preds = []
        for i in range(0, len(dense_t), batch_size):
            batch_dense = dense_t[i:i + batch_size]
            batch_sparse = sparse_t[i:i + batch_size]
            logits = model(batch_dense, batch_sparse).squeeze(-1)
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(all_preds)

    return _predictor


def main() -> None:
    cfg = _load_cfg()
    _apply_fast_overrides(cfg)
    seed = cfg.get("seed", 42)

    output_dir = Path(cfg.get("output_dir", "./models"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 特征构建/加载
    train_df, valid_df, test_df, spec = _ensure_ctr_features(seed)
    dense, sparse, vocab = spec["dense_features"], spec["sparse_features"], spec["sparse_vocab_sizes"]
    log.info(f"特征：dense={len(dense)} sparse={len(sparse)}  样本 train/valid/test={len(train_df):,}/{len(valid_df):,}/{len(test_df):,}")

    device = pick_device()
    log.info(f"训练设备：{device}")
    results: dict[str, dict] = {}

    # 2. LightGBM
    log.info("== 训练 LightGBM ==")
    lgb_cfg = cfg.rank_cfgs.lgb
    lgb_model = LGBRanker(
        objective=lgb_cfg.objective,
        num_leaves=lgb_cfg.num_leaves,
        learning_rate=lgb_cfg.learning_rate,
        n_estimators=lgb_cfg.n_estimators,
        early_stopping_rounds=lgb_cfg.early_stopping_rounds,
        random_state=seed,
    ).fit(train_df, valid_df, dense, sparse)
    lgb_model.save(str(output_dir / "lgb.txt"))
    lgb_metrics = _evaluate_on_test("LightGBM", lgb_model.predict, test_df)
    lgb_fi = lgb_model.feature_importance().head(10).to_dicts()
    results["LightGBM"] = {**lgb_metrics, "feature_importance_top10": lgb_fi}

    # 3. DeepFM
    log.info("== 训练 DeepFM ==")
    dfm_cfg = cfg.rank_cfgs.deepfm
    deepfm = DeepFM(
        dense_dim=len(dense),
        sparse_vocab_sizes=vocab,
        sparse_features=sparse,
        embedding_dim=dfm_cfg.embedding_dim,
        dnn_hidden_units=list(dfm_cfg.dnn_hidden_units),
        dropout=dfm_cfg.dropout,
        l2_reg=dfm_cfg.l2_reg,
    )
    dfm_history = train_model(
        deepfm, train_df, valid_df,
        dense_features=dense, sparse_features=sparse,
        epochs=dfm_cfg.epochs, batch_size=dfm_cfg.batch_size,
        lr=dfm_cfg.lr, patience=dfm_cfg.early_stopping_patience,
        use_amp=dfm_cfg.use_amp,
        save_path=str(output_dir / "deepfm.pt"),
        random_state=seed,
        sample_ratio=cfg.get("sample_ratio", 1.0),
    )
    deepfm = deepfm.to(device)
    dfm_metrics = _evaluate_on_test(
        "DeepFM", _predict_torch(deepfm, device, dense, sparse), test_df
    )
    results["DeepFM"] = {
        **dfm_metrics,
        "best_val_auc": max(dfm_history["val_auc"]) if dfm_history["val_auc"] else 0.0,
        "n_epochs_trained": len(dfm_history["train_loss"]),
    }

    # 4. Wide & Deep
    log.info("== 训练 Wide & Deep ==")
    wd_cfg = cfg.rank_cfgs.widedeep
    widedeep = WideDeep(
        dense_dim=len(dense),
        sparse_vocab_sizes=vocab,
        sparse_features=sparse,
        embedding_dim=wd_cfg.embedding_dim,
        dnn_hidden_units=list(wd_cfg.dnn_hidden_units),
        dropout=wd_cfg.dropout,
    )
    wd_history = train_model(
        widedeep, train_df, valid_df,
        dense_features=dense, sparse_features=sparse,
        epochs=wd_cfg.epochs, batch_size=wd_cfg.batch_size,
        lr=wd_cfg.lr, patience=wd_cfg.early_stopping_patience,
        use_amp=False,
        save_path=str(output_dir / "widedeep.pt"),
        random_state=seed,
        sample_ratio=cfg.get("sample_ratio", 1.0),
    )
    widedeep = widedeep.to(device)
    wd_metrics = _evaluate_on_test(
        "Wide&Deep", _predict_torch(widedeep, device, dense, sparse), test_df
    )
    results["Wide&Deep"] = {
        **wd_metrics,
        "best_val_auc": max(wd_history["val_auc"]) if wd_history["val_auc"] else 0.0,
        "n_epochs_trained": len(wd_history["train_loss"]),
    }

    # 5. 写 benchmark JSON
    bench_path = Path("reports/rank_benchmark.json")
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    log.info(f"排序基准已写入：{bench_path}")

    # 6. 保存辅助文件（供 Streamlit 可视化使用）
    lgb_fi_path = Path("reports/lgb_feature_importance.json")
    lgb_fi_path.write_text(json.dumps(lgb_fi, ensure_ascii=False, indent=2))
    log.info(f"LightGBM 特征重要度已写入：{lgb_fi_path}")

    dfm_hist_path = Path("reports/deepfm_history.json")
    dfm_hist_path.write_text(json.dumps(dfm_history, ensure_ascii=False, indent=2))
    log.info(f"DeepFM 训练曲线已写入：{dfm_hist_path}")

    # 6. 自动回写报告/README
    try:
        from ecom_rec.utils.report_writer import fill_rank_report, update_readme_tables
        fill_rank_report(str(bench_path))
        recall_json = Path("reports/recall_benchmark.json")
        update_readme_tables(
            recall_json=str(recall_json) if recall_json.exists() else None,
            rank_json=str(bench_path),
        )
        log.info("报告与 README 已自动回填排序结果。")
    except ImportError:
        log.info("report_writer 未实现，跳过自动回填（Phase B 后启用）。")
    except Exception as e:
        log.warning(f"自动回填失败：{e}")

    log.info("排序训练流水线完成。")


if __name__ == "__main__":
    main()
