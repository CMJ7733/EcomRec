"""运行数据哨兵并落盘质量报告。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import polars as pl

from ecom_rec.quality.guardrail import render_markdown_report, run_quality_checks


def main() -> None:
    processed_dir = Path("data/processed")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    train = pl.read_parquet(processed_dir / "train.parquet")
    valid = pl.read_parquet(processed_dir / "valid.parquet")
    test = pl.read_parquet(processed_dir / "test.parquet")

    report = run_quality_checks(train, valid, test)

    json_path = reports_dir / "data_quality_report.json"
    md_path = reports_dir / "data_quality_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    md_path.write_text(render_markdown_report(report))

    print(f"数据质量报告已生成: {json_path} | {md_path}")


if __name__ == "__main__":
    main()
