from __future__ import annotations

import json
from pathlib import Path


def test_image_mapping_paths_resolvable_for_sampled_entries() -> None:
    mapping_path = Path("app/static/images/mapping.json")
    if not mapping_path.exists():
        return

    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    if not isinstance(mapping, dict) or not mapping:
        return

    # 采样前 500 条，避免测试过慢，同时覆盖常见映射格式。
    for item_id, payload in list(mapping.items())[:500]:
        if not isinstance(payload, dict):
            continue

        image_path = payload.get("image_path")
        if not image_path:
            continue

        expected_by_mapping = Path("app/static") / image_path
        fallback_local = Path("app/static/images") / f"{item_id}.jpg"

        assert expected_by_mapping.exists() or fallback_local.exists(), (
            f"item_id={item_id} 映射图缺失: {expected_by_mapping} / {fallback_local}"
        )
