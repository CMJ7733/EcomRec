"""补图模块基础测试：先写失败用例（TDD）"""
import gzip
import json
import sys

import polars as pl
import pytest

sys.path.insert(0, "src")

from ecom_rec.assets.image_bank import (
    choose_best_image_url,
    compute_local_coverage,
    download_and_save_jpg,
    pick_high_value_items,
    stream_meta_image_urls,
)


def test_choose_best_image_url_prefers_hi_res():
    """当同一条记录同时有 hi_res/large/thumb 时，应优先选择 hi_res。"""
    images = [
        {
            "thumb": "https://img.example.com/a_thumb.jpg",
            "large": "https://img.example.com/a_large.jpg",
            "hi_res": "https://img.example.com/a_hi.jpg",
        }
    ]

    assert choose_best_image_url(images) == "https://img.example.com/a_hi.jpg"


def test_choose_best_image_url_prefers_hi_res_globally_across_rows():
    """应在全体记录中先找 hi_res，再回退 large/thumb。"""
    images = [
        {
            "thumb": "https://img.example.com/a_thumb.jpg",
            "large": "https://img.example.com/a_large.jpg",
            "hi_res": "",
        },
        {
            "thumb": "https://img.example.com/b_thumb.jpg",
            "large": "",
            "hi_res": "https://img.example.com/b_hi.jpg",
        },
    ]

    assert choose_best_image_url(images) == "https://img.example.com/b_hi.jpg"


def test_compute_local_coverage(tmp_path):
    """本地覆盖率统计应按去重后的商品集合计算。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "A.jpg").write_bytes(b"ok")
    (image_dir / "B.jpg").mkdir()

    report = compute_local_coverage(["A", "A", "B", "C"], image_dir)

    assert report["covered"] == 1
    assert report["total"] == 3
    assert report["coverage"] == pytest.approx(1 / 3)


def test_pick_high_value_items_from_train():
    """高价值商品应按训练交互频次降序选取。"""
    train = pl.DataFrame(
        {
            "item_id": ["A", "A", "B", "C", "C", "C"],
        }
    )

    assert pick_high_value_items(train, top_n=2) == ["C", "A"]


def test_stream_meta_image_urls_keeps_scanning_until_url_found(tmp_path):
    """同一商品前序记录缺图时，应继续扫描后续记录。"""
    meta_path = tmp_path / "meta.jsonl.gz"
    rows = [
        {"parent_asin": "A", "images": [{"hi_res": "", "large": "", "thumb": ""}]},
        {"asin": "A", "images": [{"hi_res": "https://img.example.com/a_hi.jpg"}]},
    ]
    with gzip.open(meta_path, "wt", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    assert stream_meta_image_urls(meta_path, {"A"}) == {"A": "https://img.example.com/a_hi.jpg"}


def test_download_and_save_jpg_timeout(monkeypatch, tmp_path):
    """网络超时应返回 timeout 原因。"""
    import requests

    def _raise_timeout(*args, **kwargs):
        raise requests.Timeout("timeout")

    monkeypatch.setattr("ecom_rec.assets.image_bank.requests.get", _raise_timeout)
    ok, reason = download_and_save_jpg("https://img.example.com/a.jpg", tmp_path / "a.jpg")
    assert ok is False
    assert reason == "timeout"


def test_download_and_save_jpg_decode_error(monkeypatch, tmp_path):
    """非图片内容应返回 decode_error。"""

    class _Resp:
        content = b"not-an-image"

        def raise_for_status(self):
            return None

    def _fake_get(*args, **kwargs):
        return _Resp()

    monkeypatch.setattr("ecom_rec.assets.image_bank.requests.get", _fake_get)
    ok, reason = download_and_save_jpg("https://img.example.com/a.jpg", tmp_path / "a.jpg")
    assert ok is False
    assert reason == "decode_error"
