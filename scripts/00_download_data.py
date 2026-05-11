"""下载 Amazon Reviews 2023 Beauty 子集原始数据到 data/raw/"""
import sys
sys.path.insert(0, "src")

from ecom_rec.data.download import download_beauty
from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)

if __name__ == "__main__":
    log.info("开始下载 Amazon Reviews 2023 Beauty 数据集...")
    download_beauty()
    log.info("下载完成。")
