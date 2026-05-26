from __future__ import annotations

import polars as pl


def _profile_numeric(s: pl.Series) -> dict[str, float | None]:
    """计算数值列概况统计。"""
    total = len(s)
    null_ratio = float(s.null_count() / total) if total else 0.0
    non_null = s.drop_nulls()

    if len(non_null) == 0:
        return {
            "min": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "max": None,
            "null_ratio": null_ratio,
        }

    return {
        "min": float(non_null.min()),
        "p01": float(non_null.quantile(0.01)),
        "p50": float(non_null.quantile(0.50)),
        "p99": float(non_null.quantile(0.99)),
        "max": float(non_null.max()),
        "null_ratio": null_ratio,
    }


def run_quality_checks(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    test: pl.DataFrame,
) -> dict:
    """运行基础数据质量检查（最小实现版）。"""
    alerts: list[dict[str, str]] = []
    profiles: dict[str, dict[str, float | None]] = {}

    if "timestamp_sec" in train.columns and "timestamp_sec" in valid.columns:
        train_ts = train["timestamp_sec"].drop_nulls()
        valid_ts = valid["timestamp_sec"].drop_nulls()
        if len(train_ts) > 0 and len(valid_ts) > 0 and valid_ts.min() <= train_ts.max():
            alerts.append(
                {
                    "level": "P0",
                    "message": "时间切分异常：valid 起始时间不应早于或等于 train 结束时间。",
                }
            )

    for col in ("rating", "price", "timestamp_sec"):
        if col in train.columns:
            profiles[col] = _profile_numeric(train[col])

    p0_count = sum(1 for a in alerts if a.get("level") == "P0")
    return {
        "alerts": alerts,
        "profiles": profiles,
        "summary": {"p0_count": p0_count},
    }
