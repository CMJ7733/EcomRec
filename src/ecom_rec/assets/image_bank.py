"""补图模块：URL 选择与本地覆盖率统计。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping


def choose_best_image_url(images: Iterable[Mapping[str, object]]) -> str:
    """按全局优先级选择 URL：先全量 hi_res，再 large，最后 thumb。"""
    rows = list(images)
    for key in ("hi_res", "large", "thumb"):
        for row in rows:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def compute_local_coverage(target_item_ids: Iterable[object], image_dir: str | Path) -> dict[str, float | int]:
    """统计目标商品在本地图片目录中的去重覆盖率。"""
    image_path = Path(image_dir)
    unique_filenames: set[str] = set()
    covered = 0

    for item_id in target_item_ids:
        unique_filenames.add(f"{item_id}.jpg")

    total = len(unique_filenames)
    for filename in unique_filenames:
        if (image_path / filename).is_file():
            covered += 1

    coverage = covered / total if total else 0.0
    return {"covered": covered, "total": total, "coverage": coverage}
