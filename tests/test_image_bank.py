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
