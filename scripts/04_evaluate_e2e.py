"""端到端推荐评估：采样测试用户，跑完整推荐链路，计算 Top-K 指标。

用法：
    conda run -n data python scripts/04_evaluate_e2e.py
    conda run -n data python scripts/04_evaluate_e2e.py n_users=3000
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import polars as pl

from ecom_rec.eval.recall_metrics import hit_rate_at_k, ndcg_at_k, coverage
from ecom_rec.pipeline.multi_recall import MultiRecall
from ecom_rec.pipeline.rerank import mmr_rerank
from ecom_rec.recall import ALSRecaller, BPRRecaller, ItemCFRecaller, PopRecaller
from ecom_rec.rank.deepfm import DeepFM
from ecom_rec.rank.lgb import LGBRanker
from ecom_rec.utils.device import pick_device
from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)

FUSION_WEIGHTS = {"itemcf": 0.35, "bpr": 0.30, "als": 0.25, "pop": 0.10}


def _load_recommender_components():
    processed = Path("data/processed")
    models_dir = Path("models")

    train = pl.read_parquet(processed / "train.parquet")
    test = pl.read_parquet(processed / "test.parquet")
    user_map = pl.read_parquet(processed / "user_map.parquet")
    item_map = pl.read_parquet(processed / "item_map.parquet")

    spec = json.loads((processed / "feature_spec.json").read_text())

    user_stats = train.group_by("user_id").agg([
        pl.col("rating").mean().alias("user_avg_rating"),
        pl.len().alias("user_frequency"),
        ((pl.col("timestamp_sec").max() - pl.col("timestamp_sec").min()) / 86400.0).alias("user_active_days"),
    ])
    item_stats = train.group_by("item_id").agg([
        pl.col("rating").mean().alias("item_avg_rating"),
        pl.len().alias("item_review_count"),
    ])

    item_meta_cols = ["item_id"]
    for col in ["category", "brand", "title", "price"]:
        if col in train.columns:
            item_meta_cols.append(col)
    item_meta = train.select(item_meta_cols).unique(subset=["item_id"])

    import pickle
    pop = pickle.loads((models_dir / "pop.pkl").read_bytes())
    icf = pickle.loads((models_dir / "itemcf.pkl").read_bytes())
    with open(models_dir / "bpr.pkl", "rb") as f:
        bpr = pickle.load(f)
    with open(models_dir / "als.pkl", "rb") as f:
        als = pickle.load(f)

    multi_recall = MultiRecall([
        (icf, FUSION_WEIGHTS["itemcf"]),
        (bpr, FUSION_WEIGHTS["bpr"]),
        (als, FUSION_WEIGHTS["als"]),
        (pop, FUSION_WEIGHTS["pop"]),
    ])

    return {
        "train": train, "test": test,
        "user_map": user_map, "item_map": item_map, "spec": spec,
        "user_stats": user_stats, "item_stats": item_stats, "item_meta": item_meta,
        "multi_recall": multi_recall,
    }


def _build_rank_predictor(spec, models_dir, device):
    import torch
    from ecom_rec.rank.trainer import prepare_tensors

    dfm_path = models_dir / "deepfm.pt"
    if not dfm_path.exists():
        return None

    deepfm = DeepFM(
        dense_dim=len(spec["dense_features"]),
        sparse_vocab_sizes=spec["sparse_vocab_sizes"],
        sparse_features=spec["sparse_features"],
        embedding_dim=16,
        dnn_hidden_units=[256, 128, 64],
        dropout=0.3,
        l2_reg=1e-5,
    )
    deepfm.load_state_dict(torch.load(dfm_path, map_location=device))
    deepfm = deepfm.to(device).eval()

    @torch.no_grad()
    def predict(feat_df):
        dense_t, sparse_t, _ = prepare_tensors(feat_df, spec["dense_features"], spec["sparse_features"])
        dense_t, sparse_t = dense_t.to(device), sparse_t.to(device)
        logits = deepfm(dense_t, sparse_t).squeeze(-).cpu().numpy()
        return 1 / (1 + np.exp(-logits))

    return predict


def _build_ctr_features_for_candidates(user_id, candidates, user_stats, item_stats,
                                        user_map, item_map, item_meta, spec):
    uid_map = {r["user_id"]: r["user_idx"] for r in user_map.iter_rows(named=True)}
    iid_map = {r["item_id"]: r["item_idx"] for r in item_map.iter_rows(named=True)}

    u_stat_rows = user_stats.filter(pl.col("user_id") == user_id)
    u_stat = u_stat_rows.to_dicts()[0] if len(u_stat_rows) > 0 else {}

    i_stats_dict = {r["item_id"]: r for r in item_stats.iter_rows(named=True)}
    meta_dict = {r["item_id"]: r for r in item_meta.iter_rows(named=True)}

    rows = []
    for item_id in candidates:
        i_stat = i_stats_dict.get(item_id, {})
        meta = meta_dict.get(item_id, {})
        rows.append({
            "user_avg_rating": u_stat.get("user_avg_rating", 4.0),
            "user_frequency": float(u_stat.get("user_frequency", 5)),
            "user_active_days": u_stat.get("user_active_days", 0.0),
            "item_avg_rating": i_stat.get("item_avg_rating", 4.0),
            "item_review_count": float(i_stat.get("item_review_count", 1)),
            "item_price_quantile": 0.5,
            "user_idx": uid_map.get(user_id, 0),
            "item_idx": iid_map.get(item_id, 0),
            "category_idx": 0,
            "brand_idx": 0,
            "weekday": 0,
            "hour": 12,
            "label": 0,
        })
    return pl.DataFrame(rows)


def main():
    from omegaconf import OmegaConf
    cli = OmegaConf.from_cli()
    n_users = cli.get("n_users", 3000)

    device = pick_device()
    log.info(f"设备: {device}, 采样用户数: {n_users}")

    log.info("加载模型和数据...")
    comp = _load_recommender_components()
    predict = _build_rank_predictor(comp["spec"], Path("models"), device)

    test = comp["test"]
    train = comp["train"]
    gt = {}
    for row in test.group_by("user_id").agg(pl.col("item_id").alias("items")).iter_rows(named=True):
        gt[row["user_id"]] = list(row["items"])

    all_items = set(train["item_id"].unique().to_list()) | set(test["item_id"].unique().to_list())
    user_history = {}
    for row in train.group_by("user_id").agg(pl.col("item_id").alias("items")).iter_rows(named=True):
        user_history[row["user_id"]] = set(row["items"])

    eval_users = list(gt.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(eval_users)
    eval_users = eval_users[:n_users]
    log.info(f"评估用户: {len(eval_users):,}")

    item_categories = {}
    if "category" in comp["item_meta"].columns:
        for r in comp["item_meta"].select(["item_id", "category"]).iter_rows(named=True):
            if r["category"]:
                item_categories[r["item_id"]] = r["category"]

    recall_recs = {}
    final_recs = {}
    total = len(eval_users)

    log.info("开始端到端推荐评估...")
    t0 = time.time()
    for i, uid in enumerate(eval_users):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            log.info(f"进度 {i+1}/{total} ({elapsed/60:.1f}min, ETA {eta/60:.1f}min)")

        candidates = comp["multi_recall"].recommend(uid, k=200)
        recall_recs[uid] = candidates

        if predict is not None and candidates:
            feat_df = _build_ctr_features_for_candidates(
                uid, candidates, comp["user_stats"], comp["item_stats"],
                comp["user_map"], comp["item_map"], comp["item_meta"], comp["spec"],
            )
            scores_arr = predict(feat_df)
            scores = {iid: float(s) for iid, s in zip(candidates, scores_arr)}
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            top50 = [iid for iid, _ in ranked[:50]]
            final = mmr_rerank(top50, scores, item_categories, k=10, lambda_=0.5)
            final_recs[uid] = final
        else:
            final_recs[uid] = candidates[:10]

    elapsed = time.time() - t0
    log.info(f"评估完成，耗时 {elapsed/60:.1f}min")

    results = {}
    for k in [10, 50]:
        results[f"Recall_HR@{k}"] = hit_rate_at_k(recall_recs, gt, k)
        results[f"Recall_NDCG@{k}"] = ndcg_at_k(recall_recs, gt, k)

    results["Recall_Coverage@50"] = coverage(recall_recs, all_items, 50)

    for k in [5, 10]:
        results[f"E2E_HR@{k}"] = hit_rate_at_k(final_recs, gt, k)
        results[f"E2E_NDCG@{k}"] = ndcg_at_k(final_recs, gt, k)

    recommended_cats = []
    for uid, items in final_recs.items():
        for iid in items[:10]:
            if iid in item_categories:
                recommended_cats.append(item_categories[iid])
    if recommended_cats:
        from collections import Counter
        cat_counts = Counter(recommended_cats)
        total_recs = len(recommended_cats)
        entropy = -sum((c / total_recs) * np.log2(c / total_recs) for c in cat_counts.values())
        max_entropy = np.log2(len(cat_counts)) if len(cat_counts) > 1 else 1.0
        results["E2E_Diversity"] = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        results["E2E_Category_Count"] = len(cat_counts)
    else:
        results["E2E_Diversity"] = 0.0
        results["E2E_Category_Count"] = 0

    results["n_eval_users"] = len(eval_users)

    out_path = Path("reports/e2e_benchmark.json")
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    log.info(f"端到端评估结果: { {k: round(v, 4) if isinstance(v, float) else v for k, v in results.items()} }")
    log.info(f"已保存到 {out_path}")


if __name__ == "__main__":
    main()
