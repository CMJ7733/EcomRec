"""训练所有排序模型并评估"""
import sys
sys.path.insert(0, "src")

from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)

if __name__ == "__main__":
    log.info("排序模型训练脚本（实现见 Sprint 3）")
