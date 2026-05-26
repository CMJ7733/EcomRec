"""补图模块：URL 选择与本地覆盖率统计。"""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Iterable, Mapping

import polars as pl
import requests
from PIL import Image, UnidentifiedImageError


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


def pick_high_value_items(train: pl.DataFrame, top_n: int) -> list[str]:
    """按训练集交互频次选取高价值商品。"""
    if top_n <= 0:
        return []
    if "item_id" not in train.columns:
        return []

    ranked = (
        train.select(pl.col("item_id").cast(pl.Utf8, strict=False).alias("item_id"))
        .drop_nulls("item_id")
        .group_by("item_id")
        .agg(pl.len().alias("cnt"))
        .sort(by=["cnt", "item_id"], descending=[True, False])
        .head(top_n)
    )
    return ranked.get_column("item_id").to_list()


def stream_meta_image_urls(meta_gz_path: Path, target_item_ids: set[str]) -> dict[str, str]:
    """流式读取 meta 文件，提取目标商品的最优图片 URL。"""
    if not target_item_ids:
        return {}

    remain = set(target_item_ids)
    url_map: dict[str, str] = {}

    with gzip.open(meta_gz_path, "rt", encoding="utf-8") as fp:
        for line in fp:
            if not remain:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_id = obj.get("parent_asin") or obj.get("asin")
            if not isinstance(item_id, str) or item_id not in remain:
                continue

            images = obj.get("images")
            if not isinstance(images, list):
                images = []

            best_url = choose_best_image_url(images)
            if best_url:
                url_map[item_id] = best_url
                remain.discard(item_id)

    return url_map


def download_and_save_jpg(url: str, output_file: Path, timeout: int = 12) -> tuple[bool, str]:
    """下载图片并保存为 512px 内缩放 JPEG，返回状态与失败原因。"""
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.Timeout:
        return False, "timeout"
    except Exception:
        return False, "unknown_error"

    try:
        resp.raise_for_status()
    except requests.HTTPError:
        return False, "http_error"
    except Exception:
        return False, "unknown_error"

    try:
        with Image.open(io.BytesIO(resp.content)) as img:
            rgb = img.convert("RGB")
            rgb.thumbnail((512, 512))
            output_file.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(output_file, format="JPEG", quality=90)
        return True, "ok"
    except UnidentifiedImageError:
        return False, "decode_error"
    except OSError:
        return False, "io_error"
    except Exception:
        return False, "unknown_error"
