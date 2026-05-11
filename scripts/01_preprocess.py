"""清洗原始数据并划分 train/valid/test"""
import sys
sys.path.insert(0, "src")

from omegaconf import OmegaConf
from ecom_rec.data.clean import clean_data
from ecom_rec.data.split import split_data
from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)

if __name__ == "__main__":
    cfg = OmegaConf.load("configs/data/amazon_beauty.yaml")
    log.info("开始清洗数据...")
    clean_data(cfg)
    log.info("开始划分数据集...")
    split_data(cfg)
    log.info("预处理完成。")
