"""补图模块基础测试：先写失败用例（TDD）"""
import sys

import pytest

sys.path.insert(0, "src")

from ecom_rec.assets.image_bank import choose_best_image_url, compute_local_coverage


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


def test_compute_local_coverage(tmp_path):
    """本地覆盖率统计应正确返回 covered/total/coverage。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "A.jpg").write_bytes(b"ok")

    report = compute_local_coverage({"A", "B", "C"}, image_dir)

    assert report["covered"] == 1
    assert report["total"] == 3
    assert report["coverage"] == pytest.approx(1 / 3)
