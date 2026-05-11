"""召回模型对比页面"""
import sys
sys.path.insert(0, "src")
from pathlib import Path
import json
import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="召回模型对比", page_icon="🔍", layout="wide")
st.title("🔍 召回模型对比")
st.markdown("对比 **Top-Pop / ItemCF / BPR-MF / ALS** 四种召回策略的 HR@K / Recall@K / NDCG@K 指标。")

PROCESSED = Path("data/processed")
REPORTS = Path("reports")

@st.cache_data(show_spinner="加载召回评估结果...")
def load_recall_results():
    result_path = REPORTS / "recall_benchmark.json"
    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)
    return None

results = load_recall_results()

if results is None:
    st.warning("召回评估结果尚未生成。请先运行 `notebooks/03_recall_benchmark.ipynb`。")
    st.info("运行后会在 `reports/recall_benchmark.json` 生成评估结果。")
    # 展示示例数据（演示模式）
    st.markdown("#### 示例展示（模拟数据）")
    results = {
        "Top-Pop":  {"HR@10": 0.05, "HR@50": 0.14, "Recall@50": 0.14, "NDCG@50": 0.08, "Coverage@100": 0.02},
        "ItemCF":   {"HR@10": 0.09, "HR@50": 0.21, "Recall@50": 0.21, "NDCG@50": 0.12, "Coverage@100": 0.15},
        "BPR-MF":   {"HR@10": 0.11, "HR@50": 0.26, "Recall@50": 0.26, "NDCG@50": 0.14, "Coverage@100": 0.22},
        "ALS":      {"HR@10": 0.12, "HR@50": 0.28, "Recall@50": 0.28, "NDCG@50": 0.15, "Coverage@100": 0.24},
    }

# ---- 指标选择 ----
col1, col2 = st.columns([1, 3])
with col1:
    k_option = st.select_slider("选择 K 值", options=[10, 50, 100], value=50)
    metric = st.radio("主要指标", ["HR", "Recall", "NDCG"])
    metric_key = f"{metric}@{k_option}"

# 构建 DataFrame
rows = []
for model_name, metrics in results.items():
    for key, val in metrics.items():
        rows.append({"模型": model_name, "指标": key, "值": val})
df_results = pd.DataFrame(rows)

with col2:
    # 主指标柱状图
    df_main = df_results[df_results["指标"] == metric_key]
    fig = px.bar(
        df_main, x="模型", y="值", color="模型", text="值",
        title=f"{metric_key} 对比",
        color_discrete_sequence=["#FF9800", "#2196F3", "#4CAF50", "#9C27B0"],
        template="plotly_white", height=400,
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- 雷达图：多维度综合对比 ----
st.markdown("### 多维度综合对比（雷达图）")
radar_metrics = [f"HR@{k_option}", f"Recall@{k_option}", f"NDCG@{k_option}", f"Coverage@100"]
fig_radar = go.Figure()
colors = ["#FF9800", "#2196F3", "#4CAF50", "#9C27B0"]
for i, (model_name, metrics) in enumerate(results.items()):
    vals = [metrics.get(m, 0.0) for m in radar_metrics]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=radar_metrics + [radar_metrics[0]],
        fill="toself",
        name=model_name,
        line_color=colors[i % len(colors)],
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, max(
        max(results[m].get(k, 0) for m in results) for k in radar_metrics
    ) * 1.2])),
    template="plotly_white", height=450,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ---- 完整指标表 ----
st.markdown("### 完整评估指标表")
table_data = {model: results[model] for model in results}
df_table = pd.DataFrame(table_data).T.round(4)
st.dataframe(df_table, use_container_width=True)
