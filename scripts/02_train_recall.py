"""训练四路召回模型 + MultiRecall 融合，评估并落盘。

用法：
    python scripts/02_train_recall.py                 # 默认（生产模式）
    python scripts/02_train_recall.py fast=true       # 快速模式（支持M 芯片本地）
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "src")

import polars as pl
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from ecom_rec.eval.recall_metrics import evaluate_recall
from ecom_rec.pipeline.multi_recall import MultiRecall
from ecom_rec.recall import ALSRecaller, BPRRecaller, ItemCFRecaller, PopRecaller
from ecom_rec.utils.io import read_parquet
from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)

# ADR-4 加权融合权重
FUSION_WEIGHTS = {"itemcf": 0.35, "bpr": 0.30, "als": 0.25, "pop": 0.10}
EVAL_K_LIST = [10, 50, 100]
EVAL_MAX_USERS_FAST = 5000


def _load_or_train(path: Path, recaller, train_data, label: str):
    """pkl 已存在则加载，否则训练 + 保存"""
    if path.exists():
        log.info(f"加载已训练模型：{path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    log.info(f"== 训练 {label} ==")
    model = recaller.fit(train_data)
    _persist(model, path)
    return model


def _load_cfg() -> OmegaConf:
    """加载主配置 + CLI override（如 fast=true）"""
    root = OmegaConf.load("configs/config.yaml")
    recall_cfgs = {
        "pop": OmegaConf.load("configs/recall/pop.yaml"),
        "itemcf": OmegaConf.load("configs/recall/itemcf.yaml"),
        "bpr": OmegaConf.load("configs/recall/bpr.yaml"),
        "als": OmegaConf.load("configs/recall/als.yaml"),
    }
    cli = OmegaConf.from_cli()
    return OmegaConf.merge(root, {"recall_cfgs": recall_cfgs}, cli)


def _apply_fast_overrides(cfg: OmegaConf) -> None:
    """fast 模式：缩减迭代次数，加快本地实验"""
    if not cfg.get("fast", False):
        return
    log.info("[fast] 模式开启：缩减 ALS/BPR 迭代次数")
    cfg.recall_cfgs.als.iterations = 20
    cfg.recall_cfgs.bpr.iterations = 30


def _build_ground_truth(valid: pl.DataFrame) -> dict[str, list[str]]:
    """从 valid 构建 {user_id: [item_id, ...]} 真值字典"""
    gt = {}
    for row in valid.group_by("user_id").agg(pl.col("item_id").alias("items")).iter_rows(named=True):
        gt[row["user_id"]] = list(row["items"])
    return gt


def _persist(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"模型已保存：{path}")


def _evaluate(
    recaller,
    eval_users: list[str],
    ground_truth: dict[str, list[str]],
    all_items: set[str],
    k_list: list[int],
    label: str,
) -> dict[str, float]:
    """对单个 recaller 在评估用户集合上跑批量推荐 + 评估（带进度条）"""
    log.info(f"评估 {label} ...")
    max_k = max(k_list)
    recs: dict[str, list[str]] = {}
    for uid in tqdm(eval_users, desc=f"recommend[{label}]", unit="user"):
        recs[uid] = recaller.recommend(uid, k=max_k)
    metrics = evaluate_recall(recs, ground_truth, all_items, k_list=k_list)
    log.info(f"{label} 评估结果：{ {k: round(v, 4) for k, v in metrics.items()} }")
    return metrics


def main() -> None:
    cfg = _load_cfg()
    _apply_fast_overrides(cfg)

    seed = cfg.get("seed", 42)
    output_dir = Path(cfg.get("output_dir", "./models"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    processed = Path("data/processed")
    log.info("加载 train/valid ...")
    train = read_parquet(processed / "train.parquet")
    valid = read_parquet(processed / "valid.parquet")
    log.info(f"train={len(train):,}  valid={len(valid):,}")

    all_items = set(train["item_id"].unique().to_list()) | set(valid["item_id"].unique().to_list())

    # 2. 真值 + 评估用户集合
    gt = _build_ground_truth(valid)
    eval_users = list(gt.keys())
    if cfg.get("fast", False) and len(eval_users) > EVAL_MAX_USERS_FAST:
        rng = pl.DataFrame({"u": eval_users}).sample(n=EVAL_MAX_USERS_FAST, seed=seed)
        eval_users = rng["u"].to_list()
        log.info(f"[fast] 评估用户采样到 {len(eval_users):,}")
    else:
        log.info(f"评估用户数：{len(eval_users):,}")

    # 3. 训练四路召回
    recallers: dict[str, object] = {}

    recallers["pop"] = _load_or_train(output_dir / "pop.pkl", PopRecaller(), train, "Top-Pop")

    icf_cfg = cfg.recall_cfgs.itemcf
    recallers["itemcf"] = _load_or_train(
        output_dir / "itemcf.pkl",
        ItemCFRecaller(n_neighbors=icf_cfg.n_neighbors, use_iuf=icf_cfg.get("iuf_weight", True)),
        train, "ItemCF",
    )

    bpr_cfg = cfg.recall_cfgs.bpr
    recallers["bpr"] = _load_or_train(
        output_dir / "bpr.pkl",
        BPRRecaller(factors=bpr_cfg.factors, iterations=bpr_cfg.iterations, learning_rate=bpr_cfg.learning_rate, regularization=bpr_cfg.regularization, random_state=seed),
        train, "BPR-MF",
    )

    als_cfg = cfg.recall_cfgs.als
    recallers["als"] = _load_or_train(
        output_dir / "als.pkl",
        ALSRecaller(factors=als_cfg.factors, iterations=als_cfg.iterations, regularization=als_cfg.regularization, random_state=seed),
        train, "ALS",
    )

    # 4. 评估单路
    results: dict[str, dict[str, float]] = {}
    name_map = {"pop": "Top-Pop", "itemcf": "ItemCF", "bpr": "BPR-MF", "als": "ALS"}
    for key, recaller in recallers.items():
        results[name_map[key]] = _evaluate(recaller, eval_users, gt, all_items, EVAL_K_LIST, name_map[key])

    # 5. MultiRecall 融合评估
    log.info("== 构建 MultiRecall（加权融合）==")
    multi = MultiRecall([
        (recallers["itemcf"], FUSION_WEIGHTS["itemcf"]),
        (recallers["bpr"], FUSION_WEIGHTS["bpr"]),
        (recallers["als"], FUSION_WEIGHTS["als"]),
        (recallers["pop"], FUSION_WEIGHTS["pop"]),
    ])
    results["MultiRecall"] = _evaluate(multi, eval_users, gt, all_items, EVAL_K_LIST, "MultiRecall")

    # 6. 写 benchmark JSON
    bench_path = Path("reports/recall_benchmark.json")
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    log.info(f"召回基准已写入：{bench_path}")

    # 7. 自动回写报告/README（report_writer 见 Phase B）
    try:
        from ecom_rec.utils.report_writer import fill_recall_report, update_readme_tables
        fill_recall_report(str(bench_path))
        update_readme_tables(recall_json=str(bench_path), rank_json=None)
        log.info("报告与 README 已自动回填召回结果。")
    except ImportError:
        log.info("report_writer 未实现，跳过自动回填（Phase B 后启用）。")
    except Exception as e:
        log.warning(f"自动回填失败：{e}")

    log.info("召回训练流水线完成。")


if __name__ == "__main__":
    main()
