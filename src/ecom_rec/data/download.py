"""从 Amazon Reviews 2023 下载 Beauty 子集数据到 data/raw/"""
import gzip
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)

# Amazon Reviews 2023 官方下载地址
REVIEW_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Beauty_and_Personal_Care.jsonl.gz"
META_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl.gz"


def _download_file(url: str, dest: Path) -> None:
    """下载单个文件，带进度条"""
    if dest.exists():
        log.info(f"文件已存在，跳过下载：{dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"正在下载：{url}")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    log.info(f"下载完成：{dest}")


def download_beauty(raw_dir: str = "data/raw") -> None:
    """下载 Beauty 评论数据与商品元数据"""
    raw = Path(raw_dir)
    _download_file(REVIEW_URL, raw / "Beauty_and_Personal_Care.jsonl.gz")
    _download_file(META_URL, raw / "meta_Beauty_and_Personal_Care.jsonl.gz")
    log.info("所有数据文件下载完成。")
