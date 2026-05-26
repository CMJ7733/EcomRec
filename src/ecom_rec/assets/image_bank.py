"""补图模块：URL 选择与本地覆盖率统计。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping


def choose_best_image_url(images: Iterable[Mapping[str, object]]) -> str:
    """按 hi_res -> large -> thumb 的优先级选择首个可用 URL。"""
    for row in images:
        for key in ("hi_res", "large", "thumb"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def compute_local_coverage(target_item_ids: Iterable[object], image_dir: str | Path) -> dict[str, float | int]:
    """统计目标商品在本地图片目录中的覆盖率。"""
    image_path = Path(image_dir)
    total = 0
    covered = 0

    for item_id in target_item_ids:
        total += 1
        if (image_path / f"{item_id}.jpg").exists():
            covered += 1

    coverage = covered / total if total else 0.0
    return {"covered": covered, "total": total, "coverage": coverage}
