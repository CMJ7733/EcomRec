"""排序模型分析页面"""
import sys
sys.path.insert(0, "src")
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="排序模型分析", page_icon="🎯", layout="wide")
st.title("🎯 精排模型分析")
st.markdown("对比 **LightGBM / DeepFM / Wide&Deep** 三种精排模型的 AUC / LogLoss / GAUC。")

REPORTS = Path("reports")

@st.cache_data(show_spinner="加载排序评估结果...")
def load_rank_results():
    path = REPORTS / "rank_benchmark.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

rank_results = load_rank_results()

if rank_results is None:
    st.warning("排序评估结果尚未生成。请先运行 `notebooks/04_rank_benchmark.ipynb`。")
    # 演示模式
    rank_results = {
        "LightGBM": {"AUC": 0.738, "LogLoss": 0.438, "GAUC": 0.71, "best_val_auc": 0.742},
        "DeepFM":   {"AUC": 0.762, "LogLoss": 0.421, "GAUC": 0.74, "best_val_auc": 0.768},
        "Wide&Deep":{"AUC": 0.749, "LogLoss": 0.429, "GAUC": 0.73, "best_val_auc": 0.755},
    }

# ---- AUC 对比 ----
st.markdown("### 模型 AUC 对比")
df_rank = pd.DataFrame([
    {"模型": m, "验证集 AUC": v.get("best_val_auc", v.get("val_auc", 0)), "测试集 AUC": v.get("AUC", v.get("test_auc", 0)),
     "LogLoss": v.get("LogLoss", v.get("val_logloss", 0)), "GAUC": v.get("GAUC", v.get("test_gauc", 0))}
    for m, v in rank_results.items()
])

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("验证集 AUC", "LogLoss（越低越好）"))
colors = ["#FF9800", "#2196F3", "#4CAF50"]
for i, (_, row) in enumerate(df_rank.iterrows()):
    fig.add_trace(go.Bar(name=row["模型"], x=[row["模型"]], y=[row["验证集 AUC"]],
                         marker_color=colors[i], showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(name=row["模型"], x=[row["模型"]], y=[row["LogLoss"]],
                         marker_color=colors[i], showlegend=True), row=1, col=2)
fig.update_layout(template="plotly_white", height=400, title_text="精排模型性能对比")
st.plotly_chart(fig, use_container_width=True)

# ---- 完整指标表 ----
st.markdown("### 完整评估指标")
st.dataframe(df_rank.set_index("模型").round(4), use_container_width=True)

st.divider()

# ---- 训练曲线（如有） ----
st.markdown("### 训练曲线（DeepFM）")
history_path = REPORTS / "deepfm_history.json"
if history_path.exists():
    with open(history_path) as f:
        history = json.load(f)
    fig_h = make_subplots(rows=1, cols=2, subplot_titles=("训练 Loss", "验证 AUC"))
    fig_h.add_trace(go.Scatter(y=history["train_loss"], name="Train Loss",
                                line=dict(color="#F44336")), row=1, col=1)
    fig_h.add_trace(go.Scatter(y=history["val_auc"], name="Val AUC",
                                line=dict(color="#2196F3")), row=1, col=2)
    fig_h.update_layout(template="plotly_white", height=380)
    st.plotly_chart(fig_h, use_container_width=True)
else:
    st.info("训练曲线数据未找到（`reports/deepfm_history.json`）。运行 Notebook 04 后刷新。")

# ---- 特征重要度（LightGBM） ----
st.divider()
st.markdown("### LightGBM 特征重要度")
fi_path = REPORTS / "lgb_feature_importance.json"
if fi_path.exists():
    with open(fi_path) as f:
        fi_data = json.load(f)
    fi_df = pd.DataFrame(fi_data).sort_values("importance", ascending=False).head(15)
    fi_df["importance_pct"] = fi_df["importance"] / fi_df["importance"].sum() * 100
    fig_fi = px.bar(fi_df, x="importance_pct", y="feature", orientation="h",
                    title="特征重要度（Gain 贡献率 %）",
                    labels={"importance_pct": "Gain 贡献率 (%)", "feature": ""},
                    color="feature",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    template="plotly_white", height=500)
    fig_fi.update_layout(yaxis=dict(categoryorder="total ascending"), showlegend=False)
    fig_fi.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    st.plotly_chart(fig_fi, use_container_width=True)
else:
    st.info("特征重要度数据未找到。运行 Notebook 04 后刷新。")
