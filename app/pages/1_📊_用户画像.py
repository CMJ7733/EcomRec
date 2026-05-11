"""用户画像分析页面"""
import sys
sys.path.insert(0, "src")
from pathlib import Path
import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.set_page_config(page_title="用户画像", page_icon="📊", layout="wide")
st.title("📊 用户画像分析")
st.markdown("基于 **RFM 模型**的 KMeans 聚类，识别高价值/潜力/沉睡/流失四类用户群体。")

PROCESSED = Path("data/processed")

@st.cache_data(show_spinner="正在加载用户画像数据...")
def load_rfm():
    rfm_path = PROCESSED / "rfm_labeled.parquet"
    if rfm_path.exists():
        return pl.read_parquet(rfm_path)
    return None

rfm = load_rfm()

if rfm is None:
    st.warning("用户画像数据尚未生成。请先运行：`python notebooks/02_rfm_user_profile.ipynb` 或相关训练脚本。")
    st.info("快速启动：运行 `make data` 完成数据预处理后，打开 `notebooks/02_rfm_user_profile.ipynb` 执行全部 Cell。")
    st.stop()

rfm_pd = rfm.to_pandas()
segments = sorted(rfm_pd["user_segment"].unique().tolist())

# ---- 概览卡片 ----
st.markdown("### 数据概览")
cols = st.columns(len(segments))
for i, seg in enumerate(segments):
    cnt = (rfm_pd["user_segment"] == seg).sum()
    pct = cnt / len(rfm_pd) * 100
    cols[i].metric(seg, f"{cnt:,} 人", f"{pct:.1f}%")

st.divider()

# ---- 用户群体选择 ----
col_left, col_right = st.columns([1, 3])
with col_left:
    selected_segs = st.multiselect("选择展示的用户群体", segments, default=segments)
    k_slider = st.slider("Recency 截断天数（用于可视化）", 0, 365, 200)

filtered = rfm_pd[rfm_pd["user_segment"].isin(selected_segs)]
filtered = filtered[filtered["recency_days"] <= k_slider]

COLOR_MAP = {
    "核心高价值用户": "#2196F3",
    "潜力用户": "#4CAF50",
    "沉睡用户": "#FF9800",
    "流失用户": "#F44336",
}

with col_right:
    # 3D 散点图
    fig = px.scatter_3d(
        filtered.head(5000),  # 限制渲染点数
        x="recency_days", y="frequency", z="monetary",
        color="user_segment",
        title="RFM 用户分群 3D 可视化",
        labels={"recency_days": "近度(天)", "frequency": "频度", "monetary": "价值"},
        opacity=0.6,
        color_discrete_map=COLOR_MAP,
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- 雷达图 ----
st.markdown("### 各群体 RFM 特征对比（归一化雷达图）")
segment_stats = (
    rfm_pd.groupby("user_segment")[["recency_days", "frequency", "monetary"]]
    .mean().reset_index()
)
scaler = MinMaxScaler()
cols_to_scale = ["recency_days", "frequency", "monetary"]
segment_stats[cols_to_scale] = scaler.fit_transform(segment_stats[cols_to_scale])
segment_stats["recency_days"] = 1 - segment_stats["recency_days"]  # 反转：近度越小越好

categories = ["近度(反转)", "频度", "价值"]
fig_radar = go.Figure()
for _, row in segment_stats.iterrows():
    vals = [row["recency_days"], row["frequency"], row["monetary"]]
    color = COLOR_MAP.get(row["user_segment"], "#888")
    fig_radar.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=row["user_segment"],
        line_color=color,
        opacity=0.7,
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    template="plotly_white", height=450,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ---- 详细统计表 ----
st.markdown("### 各群体详细统计")
stats_table = (
    rfm_pd.groupby("user_segment")
    .agg(
        用户数=("user_id", "count"),
        平均近度_天=("recency_days", "mean"),
        平均频度=("frequency", "mean"),
        平均价值=("monetary", "mean"),
    )
    .round(2)
    .reset_index()
    .rename(columns={"user_segment": "用户群体"})
)
st.dataframe(stats_table, use_container_width=True)
