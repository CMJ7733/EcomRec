"""Streamlit Dashboard 主入口（增强版）"""
import json
import sys
from pathlib import Path

import polars as pl
import streamlit as st
from _theme import apply_theme

sys.path.insert(0, "src")

st.set_page_config(
    page_title="EcomRec — 电商推荐系统",
    page_icon="🛍️",
    layout="wide",
)
apply_theme()

st.title("🛍️ EcomRec — 电商用户行为分析与深度混合推荐系统")
st.markdown("""
基于 **Amazon Reviews 2023 Beauty** 子集，完整复现工业界"召回-精排"双塔推荐架构。
""")

REPORTS = Path("reports")
PROCESSED = Path("data/processed")

# 预留：数据质量与补图覆盖报告路径（后续看板接入）
quality_path = REPORTS / "data_quality_report.json"
coverage_path = REPORTS / "image_coverage_report.json"

# 轻量状态占位：确保关键报告路径可见且被实际使用
quality_ready = quality_path.exists()
coverage_ready = coverage_path.exists()
st.caption(
    "资产状态 · 数据质量报告："
    f"{'已就绪' if quality_ready else '未生成'}"
    " · 图片覆盖报告："
    f"{'已就绪' if coverage_ready else '未生成'}"
)

# ---- 关键数字概览 ----
st.markdown("### 📊 系统概览")
col1, col2, col3, col4, col5 = st.columns(5)

data_stats = {}
if (PROCESSED / "train.parquet").exists():
    train = pl.read_parquet(PROCESSED / "train.parquet")
    data_stats["用户数"] = f"{train['user_id'].n_unique():,}"
    data_stats["商品数"] = f"{train['item_id'].n_unique():,}"
    data_stats["交互数"] = f"{len(train):,}"

recall_stats = {}
recall_path = REPORTS / "recall_benchmark.json"
if recall_path.exists():
    with open(recall_path) as f:
        rb = json.load(f)
    if "MultiRecall" in rb:
        recall_stats["融合HR@50"] = f"{rb['MultiRecall'].get('HR@50', 0):.2%}"
    if "ALS" in rb:
        recall_stats["ALS HR@50"] = f"{rb['ALS'].get('HR@50', 0):.2%}"

rank_stats = {}
rank_path = REPORTS / "rank_benchmark.json"
if rank_path.exists():
    with open(rank_path) as f:
        rkb = json.load(f)
    if "DeepFM" in rkb:
        rank_stats["DeepFM AUC"] = f"{rkb['DeepFM'].get('AUC', 0):.4f}"

with col1:
    st.metric("独立用户", data_stats.get("用户数", "-"))
with col2:
    st.metric("独立商品", data_stats.get("商品数", "-"))
with col3:
    st.metric("交互记录", data_stats.get("交互数", "-"))
with col4:
    st.metric("融合 HR@50", recall_stats.get("融合HR@50", "-"))
with col5:
    st.metric("DeepFM AUC", rank_stats.get("DeepFM AUC", "-"))

st.divider()

# ---- 系统架构流程图 ----
st.markdown("### 🏗️ 系统架构")

ARCH_HTML = """
<style>
.arch-container {
    max-width: 820px;
    margin: 0 auto;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
.arch-layer {
    border-radius: 16px;
    padding: 14px 20px;
    margin-bottom: 6px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    display: flex;
    align-items: center;
    gap: 12px;
}
.arch-layer-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
    min-width: 80px;
    color: rgba(0,0,0,0.65);
}
.arch-layer-desc {
    font-size: 13px;
    color: rgba(0,0,0,0.5);
}
.arch-arrow {
    text-align: center;
    color: rgba(0,0,0,0.2);
    font-size: 18px;
    line-height: 1;
    margin: 2px 0;
}
.arch-modules {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: center;
    margin-bottom: 6px;
}
.arch-mod {
    border-radius: 12px;
    padding: 10px 14px;
    background: white;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    text-align: center;
    min-width: 90px;
}
.arch-mod-name {
    font-size: 13px;
    font-weight: 600;
    color: rgba(0,0,0,0.8);
}
.arch-mod-desc {
    font-size: 11px;
    color: rgba(0,0,0,0.4);
    margin-top: 2px;
}
.arch-output {
    border-radius: 16px;
    padding: 16px 24px;
    text-align: center;
    font-size: 15px;
    font-weight: 600;
    color: white;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    box-shadow: 0 4px 20px rgba(102,126,234,0.3);
}
</style>
<div class="arch-container">

<div class="arch-layer" style="background:rgba(142,142,147,0.08);">
    <div class="arch-layer-title">📊 原始数据</div>
    <div class="arch-layer-desc">Amazon Reviews 2023 · Beauty · 598万交互 · 66万用户 · 25万商品</div>
</div>
<div class="arch-arrow">↓</div>

<div class="arch-layer" style="background:rgba(0,122,255,0.08);">
    <div class="arch-layer-title">🔧 数据治理</div>
    <div class="arch-layer-desc">K-core(5) 过滤 → 时间戳解析 → 时间序 80/10/10 切分 + LOO</div>
</div>
<div class="arch-arrow">↓</div>

<div class="arch-layer" style="background:rgba(52,199,89,0.08);">
    <div class="arch-layer-title">👤 用户画像</div>
    <div class="arch-layer-desc">RFM 建模 (Recency / Frequency / Monetary) → KMeans(k=4) → 4类群体画像</div>
</div>
<div class="arch-arrow">↓</div>

<div class="arch-layer" style="background:rgba(52,199,89,0.12);">
    <div class="arch-layer-title">🔍 召回层</div>
    <div class="arch-layer-desc">四路并行召回 → 加权位置衰减融合 → Top-200 候选</div>
</div>

<div class="arch-modules">
    <div class="arch-mod" style="border-left:3px solid rgba(255,149,0,0.7);">
        <div class="arch-mod-name">Top-Pop</div>
        <div class="arch-mod-desc">热门兜底</div>
    </div>
    <div class="arch-mod" style="border-left:3px solid rgba(0,122,255,0.7);">
        <div class="arch-mod-name">ItemCF</div>
        <div class="arch-mod-desc">协同过滤</div>
    </div>
    <div class="arch-mod" style="border-left:3px solid rgba(52,199,89,0.7);">
        <div class="arch-mod-name">BPR-MF</div>
        <div class="arch-mod-desc">矩阵分解</div>
    </div>
    <div class="arch-mod" style="border-left:3px solid rgba(175,82,222,0.7);">
        <div class="arch-mod-name">ALS</div>
        <div class="arch-mod-desc">隐语义模型</div>
    </div>
</div>
<div class="arch-arrow">↓</div>

<div class="arch-layer" style="background:rgba(255,149,0,0.08);">
    <div class="arch-layer-title">🎯 精排层</div>
    <div class="arch-layer-desc">CTR 特征工程 (6 Dense + 6 Sparse) → 深度排序 → Top-50</div>
</div>

<div class="arch-modules">
    <div class="arch-mod" style="border-left:3px solid rgba(142,142,147,0.7);">
        <div class="arch-mod-name">LightGBM</div>
        <div class="arch-mod-desc">Baseline</div>
    </div>
    <div class="arch-mod" style="border-left:3px solid rgba(255,149,0,0.7);">
        <div class="arch-mod-name">DeepFM</div>
        <div class="arch-mod-desc">FM + DNN</div>
    </div>
    <div class="arch-mod" style="border-left:3px solid rgba(0,122,255,0.7);">
        <div class="arch-mod-name">Wide&Deep</div>
        <div class="arch-mod-desc">Wide + DNN</div>
    </div>
</div>
<div class="arch-arrow">↓</div>

<div class="arch-layer" style="background:rgba(175,82,222,0.08);">
    <div class="arch-layer-title">✨ 后处理</div>
    <div class="arch-layer-desc">MMR 打散 · λ=0.5 · 类目多样性保证</div>
</div>
<div class="arch-arrow">↓</div>

<div class="arch-output">✨ 个性化 Top-10 推荐</div>

</div>
"""

st.markdown(ARCH_HTML, unsafe_allow_html=True)

st.divider()

# ---- 模块导航 ----
st.markdown("### 🧭 模块导航")
nav1, nav2, nav3, nav4 = st.columns(4)
with nav1:
    st.info("📊 **用户画像**\n\nRFM 建模 + KMeans 聚类")
with nav2:
    st.info("🔍 **召回模型对比**\n\nTop-Pop / ItemCF / BPR / ALS")
with nav3:
    st.info("🎯 **排序模型分析**\n\nLGB / DeepFM / Wide&Deep")
with nav4:
    st.info("🛍️ **推荐演示**\n\n完整链路 Top-10 推荐")

st.markdown("---")

if st.button("🚀 立即体验推荐演示", type="primary", use_container_width=True):
    st.switch_page("pages/4_🛍️_推荐演示.py")

st.caption("数据挖掘课程大作业 · 以资深工程师视角复现工业级推荐系统")
