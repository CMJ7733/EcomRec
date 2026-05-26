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


def _add_price_outlier_alert(train: pl.DataFrame, alerts: list[dict[str, str]]) -> None:
    """检测价格异常值占比并在超阈值时追加 P1 告警。"""
    if "price" not in train.columns:
        return

    price = train["price"].drop_nulls()
    if len(price) == 0:
        return

    p99 = price.quantile(0.99)
    if p99 is None:
        return

    threshold = float(p99) * 3.0
    outlier_count = int((price > threshold).sum())
    outlier_ratio = float(outlier_count / len(price))

    if outlier_ratio > 0.005:
        alerts.append(
            {
                "level": "P1",
                "message": f"价格异常值占比偏高: {outlier_ratio:.2%}",
            }
        )


def render_markdown_report(report: dict) -> str:
    """把质量检查结果渲染为 Markdown 报告。"""
    summary = report.get("summary", {})
    p0_count = summary.get("p0_count", 0)
    p1_count = summary.get("p1_count", 0)
    p2_count = summary.get("p2_count", 0)
    alerts = report.get("alerts", [])

    lines = [
        "# 数据质量报告",
        "",
        f"- P0 告警数: {p0_count}",
        f"- P1 告警数: {p1_count}",
        f"- P2 告警数: {p2_count}",
        "",
        "## 告警明细",
    ]

    if not alerts:
        lines.append("- 无告警")
        return "\n".join(lines) + "\n"

    for alert in alerts:
        level = alert.get("level", "P2")
        message = alert.get("message", "")
        lines.append(f"- [{level}] {message}")

    return "\n".join(lines) + "\n"


def run_quality_checks(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    test: pl.DataFrame,
) -> dict:
    """运行基础数据质量检查。"""
    _ = test  # 预留给后续 test 集质量规则

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

    _add_price_outlier_alert(train, alerts)

    for col in ("rating", "price", "timestamp_sec"):
        if col in train.columns:
            profiles[col] = _profile_numeric(train[col])

    p0_count = sum(1 for a in alerts if a.get("level") == "P0")
    p1_count = sum(1 for a in alerts if a.get("level") == "P1")
    p2_count = sum(1 for a in alerts if a.get("level") == "P2")
    return {
        "alerts": alerts,
        "profiles": profiles,
        "summary": {
            "p0_count": p0_count,
            "p1_count": p1_count,
            "p2_count": p2_count,
        },
    }
