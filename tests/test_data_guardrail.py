"""数据哨兵测试：先写失败用例（TDD）"""
import sys

import polars as pl

sys.path.insert(0, "src")

from ecom_rec.quality.guardrail import run_quality_checks


def test_guardrail_detects_time_leakage_p0():
    """valid 时间早于/等于 train 末尾时，应触发 P0 时间切分告警"""
    train = pl.DataFrame(
        {
            "timestamp_sec": [100, 200],
            "rating": [4.0, 5.0],
            "price": [10.0, 20.0],
        }
    )
    valid = pl.DataFrame(
        {
            "timestamp_sec": [150],
            "rating": [3.0],
            "price": [15.0],
        }
    )
    test = pl.DataFrame(
        {
            "timestamp_sec": [300],
            "rating": [4.0],
            "price": [30.0],
        }
    )

    report = run_quality_checks(train, valid, test)
    p0_messages = [a.get("message", "") for a in report["alerts"] if a.get("level") == "P0"]
    assert any("时间切分" in msg for msg in p0_messages)


def test_guardrail_profiles_price_null_ratio():
    """price 画像应包含 null_ratio，且值正确"""
    train = pl.DataFrame(
        {
            "timestamp_sec": [100, 200],
            "rating": [4.0, 5.0],
            "price": [None, 20.0],
        }
    )
    valid = pl.DataFrame(
        {
            "timestamp_sec": [300],
            "rating": [3.0],
            "price": [15.0],
        }
    )
    test = pl.DataFrame(
        {
            "timestamp_sec": [400],
            "rating": [2.0],
            "price": [10.0],
        }
    )

    report = run_quality_checks(train, valid, test)
    assert "price" in report["profiles"]
    assert report["profiles"]["price"]["null_ratio"] == 0.5
