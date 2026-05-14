"""报告与 README 自动回填工具。

用 markdown 表格行匹配的方式回写实验指标，幂等可重跑。

支持：
- fill_recall_report  → reports/03_推荐模型对比报告.md 召回表
- fill_rank_report    → reports/03_推荐模型对比报告.md 排序表
- update_readme_tables → README.md 两张实验表
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ecom_rec.utils.logger import get_logger

log = get_logger(__name__)

# 模型名到 markdown 行标签的统一映射（兼容 README 与 reports 写法）
_RECALL_NAME_VARIANTS = {
    "Top-Pop": ["Top-Pop"],
    "ItemCF": ["ItemCF"],
    "BPR-MF": ["BPR-MF"],
    "ALS": ["ALS"],
    "MultiRecall": ["MultiRecall", "Multi-Recall", "多路融合"],
}
_RANK_NAME_VARIANTS = {
    "LightGBM": ["LightGBM"],
    "DeepFM": ["DeepFM"],
    "Wide&Deep": ["Wide&Deep", "Wide & Deep"],
}


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _replace_table_row(md_text: str, name_variants: list[str], new_cells: list[str]) -> tuple[str, bool]:
    """在 markdown 中找到首单元格为 (**name**|name) 的表行，替换全行。

    返回 (新文本, 是否替换成功)
    """
    for name in name_variants:
        # 匹配：行首 | 可选 ** name ** | ... 到行尾
        pattern = re.compile(
            r"^\|\s*\*{0,2}" + re.escape(name) + r"\*{0,2}\s*\|[^\n]*$",
            flags=re.MULTILINE,
        )
        replacement = "| **" + name + "** | " + " | ".join(new_cells) + " |"
        new_text, n = pattern.subn(replacement, md_text)
        if n > 0:
            return new_text, True
    return md_text, False


def fill_recall_report(
    json_path: str,
    md_path: str = "reports/03_推荐模型对比报告.md",
) -> None:
    """把 recall_benchmark.json 的指标回填到报告 1.2 节表格。

    表头列：HR@10 | HR@50 | Recall@50 | NDCG@50 | Coverage@50
    """
    md = Path(md_path)
    if not md.exists():
        log.warning(f"报告文件不存在：{md_path}")
        return

    data = json.loads(Path(json_path).read_text())
    text = md.read_text()

    for canonical, variants in _RECALL_NAME_VARIANTS.items():
        if canonical not in data:
            continue
        m = data[canonical]
        cells = [
            _fmt(m.get("HR@10")),
            _fmt(m.get("HR@50")),
            _fmt(m.get("Recall@50")),
            _fmt(m.get("NDCG@50")),
            _fmt(m.get("Coverage@100", m.get("Coverage@50"))),
        ]
        text, ok = _replace_table_row(text, variants, cells)
        if not ok:
            log.debug(f"未在报告中找到 {canonical} 行")

    md.write_text(text)
    log.info(f"召回指标已回填：{md}")


def fill_rank_report(
    json_path: str,
    md_path: str = "reports/03_推荐模型对比报告.md",
) -> None:
    """把 rank_benchmark.json 的指标回填到报告 2.2 节表格。

    表头列：AUC | LogLoss | GAUC | 训练时间
    """
    md = Path(md_path)
    if not md.exists():
        log.warning(f"报告文件不存在：{md_path}")
        return

    data = json.loads(Path(json_path).read_text())
    text = md.read_text()

    for canonical, variants in _RANK_NAME_VARIANTS.items():
        if canonical not in data:
            continue
        m = data[canonical]
        cells = [
            _fmt(m.get("AUC")),
            _fmt(m.get("LogLoss")),
            _fmt(m.get("GAUC")),
            "—",  # 训练时间留空，避免误导
        ]
        text, ok = _replace_table_row(text, variants, cells)
        if not ok:
            log.debug(f"未在报告中找到 {canonical} 行")

    md.write_text(text)
    log.info(f"排序指标已回填：{md}")


def update_readme_tables(
    recall_json: str | None = None,
    rank_json: str | None = None,
    readme_path: str = "README.md",
) -> None:
    """把 benchmark 指标回填到 README 的两张实验表。

    README 召回表头：HR@10 | HR@50 | Recall@50 | NDCG@50
    README 排序表头：AUC | LogLoss | GAUC
    """
    rd = Path(readme_path)
    if not rd.exists():
        log.warning(f"README 不存在：{readme_path}")
        return

    text = rd.read_text()

    if recall_json and Path(recall_json).exists():
        data = json.loads(Path(recall_json).read_text())
        for canonical, variants in _RECALL_NAME_VARIANTS.items():
            if canonical not in data or canonical == "MultiRecall":
                continue  # README 召回表不含 MultiRecall 行
            m = data[canonical]
            cells = [
                _fmt(m.get("HR@10")),
                _fmt(m.get("HR@50")),
                _fmt(m.get("Recall@50")),
                _fmt(m.get("NDCG@50")),
            ]
            text, _ = _replace_table_row(text, variants, cells)

    if rank_json and Path(rank_json).exists():
        data = json.loads(Path(rank_json).read_text())
        for canonical, variants in _RANK_NAME_VARIANTS.items():
            if canonical not in data:
                continue
            m = data[canonical]
            cells = [_fmt(m.get("AUC")), _fmt(m.get("LogLoss")), _fmt(m.get("GAUC"))]
            text, _ = _replace_table_row(text, variants, cells)

    rd.write_text(text)
    log.info(f"README 实验表已回填：{rd}")
