"""离线补图脚本：构建本地图片库并输出覆盖率报告。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

import polars as pl

from ecom_rec.assets.image_bank import (
    compute_local_coverage,
    download_and_save_jpg,
    pick_high_value_items,
    stream_meta_image_urls,
)


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线补图与覆盖率报告")
    parser.add_argument("--top-n", type=int, default=30000, help="按训练交互频次选取前 N 个商品")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/train.parquet"),
        help="训练集 parquet 路径",
    )
    parser.add_argument(
        "--meta-gz",
        type=Path,
        default=Path("data/raw/meta_Beauty_and_Personal_Care.jsonl.gz"),
        help="Amazon meta jsonl.gz 路径",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("app/static/images"),
        help="本地图片输出目录",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/image_coverage_report.json"),
        help="覆盖率报告输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = _build_args()

    train = pl.read_parquet(args.train)
    targets = pick_high_value_items(train, top_n=args.top_n)
    target_set = set(targets)

    url_map = stream_meta_image_urls(args.meta_gz, target_set)

    cached_hit = 0
    download_success = 0
    download_fail = 0
    failure_reasons: dict[str, int] = defaultdict(int)
    failed_items: list[str] = []

    args.image_dir.mkdir(parents=True, exist_ok=True)

    for item_id in targets:
        output_file = args.image_dir / f"{item_id}.jpg"
        if output_file.is_file():
            cached_hit += 1
            continue

        url = url_map.get(item_id, "")
        if not url:
            failure_reasons["no_url"] += 1
            failed_items.append(item_id)
            continue

        ok, reason = download_and_save_jpg(url, output_file)
        if ok:
            download_success += 1
        else:
            download_fail += 1
            failure_reasons[reason] += 1
            failed_items.append(item_id)

    coverage_info = compute_local_coverage(targets, args.image_dir)
    success = cached_hit + download_success
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_n": args.top_n,
        "train_path": str(args.train),
        "meta_gz_path": str(args.meta_gz),
        "image_dir": str(args.image_dir),
        "target_total": len(targets),
        "cached_hit": cached_hit,
        "download_success": download_success,
        "download_fail": download_fail,
        "success": success,
        "failed": len(targets) - success,
        "coverage": coverage_info["coverage"],
        "failure_reasons": dict(failure_reasons),
        "failed_sample": failed_items[:50],
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"补图完成: {success}/{len(targets)}")


if __name__ == "__main__":
    main()
