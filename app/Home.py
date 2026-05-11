"""Streamlit Dashboard 主入口"""
import streamlit as st

st.set_page_config(
    page_title="EcomRec — 电商推荐系统",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ EcomRec — 电商用户行为分析与深度混合推荐系统")
st.markdown("""
基于 **Amazon Reviews 2023 Beauty** 子集，完整复现工业界"召回-精排"双塔推荐架构。

---

### 系统模块导航
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("📊 **用户画像**\n\nRFM 建模 + KMeans 聚类")
with col2:
    st.info("🔍 **召回模型对比**\n\nTop-Pop / ItemCF / BPR / ALS")
with col3:
    st.info("🎯 **排序模型分析**\n\nLGB / DeepFM / Wide&Deep")
with col4:
    st.info("🛍️ **推荐演示**\n\n完整链路 Top-10 推荐")

st.markdown("---")
st.caption("数据挖掘课程大作业 · 以资深工程师视角复现工业级推荐系统")
