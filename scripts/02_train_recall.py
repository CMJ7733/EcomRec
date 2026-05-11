"""训练所有召回模型并评估"""
import sys
sys.path.insert(0, "src")

from omegaconf import OmegaConf
from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)

if __name__ == "__main__":
    log.info("召回模型训练脚本（实现见 Sprint 2）")
