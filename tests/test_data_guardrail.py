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


def test_guardrail_detects_price_outlier_p1():
    """price 极端异常值占比过高时，应触发 P1 告警"""
    prices = [10.0] * 99 + [9999.0]
    train = pl.DataFrame(
        {
            "timestamp_sec": list(range(100, 200)),
            "rating": [4.0] * 100,
            "price": prices,
        }
    )
    valid = pl.DataFrame(
        {
            "timestamp_sec": [600],
            "rating": [4.0],
            "price": [10.0],
        }
    )
    test = pl.DataFrame(
        {
            "timestamp_sec": [700],
            "rating": [5.0],
            "price": [20.0],
        }
    )

    report = run_quality_checks(train, valid, test)
    p1_messages = [a.get("message", "") for a in report["alerts"] if a.get("level") == "P1"]
    assert any("价格异常值占比" in msg for msg in p1_messages)


def test_guardrail_no_time_leakage_alert_when_per_user_order_ok():
    """有 user_id 时应按用户检查，避免全局最小/最大造成误报。"""
    train = pl.DataFrame(
        {
            "user_id": [1, 1, 2],
            "timestamp_sec": [100, 200, 1000],
            "rating": [4.0, 5.0, 3.0],
            "price": [10.0, 12.0, 20.0],
        }
    )
    valid = pl.DataFrame(
        {
            "user_id": [1, 2, 3],
            "timestamp_sec": [201, 1500, 50],  # user3 不重叠，不应参与泄露判定
            "rating": [4.0, 3.0, 5.0],
            "price": [11.0, 21.0, 9.0],
        }
    )
    test = pl.DataFrame(
        {
            "user_id": [1],
            "timestamp_sec": [2000],
            "rating": [5.0],
            "price": [30.0],
        }
    )

    report = run_quality_checks(train, valid, test)
    p0_messages = [a.get("message", "") for a in report["alerts"] if a.get("level") == "P0"]
    assert not any("时间切分" in msg for msg in p0_messages)


def test_guardrail_price_outlier_ratio_boundary():
    """异常值占比边界：==0.005 不告警，>0.005 告警。"""
    train_equal = pl.DataFrame(
        {
            "timestamp_sec": list(range(100, 300)),
            "rating": [4.0] * 200,
            "price": [10.0] * 199 + [1000.0],  # 1/200 == 0.005
        }
    )
    valid = pl.DataFrame(
        {
            "timestamp_sec": [10000],
            "rating": [4.0],
            "price": [10.0],
        }
    )
    test = pl.DataFrame(
        {
            "timestamp_sec": [11000],
            "rating": [5.0],
            "price": [20.0],
        }
    )

    report_equal = run_quality_checks(train_equal, valid, test)
    p1_messages_equal = [a.get("message", "") for a in report_equal["alerts"] if a.get("level") == "P1"]
    assert not any("价格异常值占比" in msg for msg in p1_messages_equal)

    train_higher = train_equal.with_columns(
        pl.Series("price", [10.0] * 198 + [1000.0, 1000.0])  # 2/200 == 0.01
    )
    report_higher = run_quality_checks(train_higher, valid, test)
    p1_messages_higher = [a.get("message", "") for a in report_higher["alerts"] if a.get("level") == "P1"]
    assert any("价格异常值占比" in msg for msg in p1_messages_higher)
