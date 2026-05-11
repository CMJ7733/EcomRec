"""推荐演示页面：端到端推荐链路展示"""
import sys
sys.path.insert(0, "src")
from pathlib import Path
import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import json

st.set_page_config(page_title="推荐演示", page_icon="🛍️", layout="wide")
st.title("🛍️ 个性化推荐演示")
st.markdown(
    "输入用户 ID，查看完整推荐链路：**多路召回（200）→ DeepFM 精排（50）→ MMR 打散（10）**"
)

PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")

@st.cache_resource(show_spinner="正在加载推荐引擎（首次加载约 10-30 秒）...")
def load_recommender():
    """加载推荐引擎（带完整错误处理）"""
    try:
        import torch
        from ecom_rec.recall.pop import PopRecaller
        from ecom_rec.recall.itemcf import ItemCFRecaller
        from ecom_rec.pipeline.multi_recall import MultiRecall
        from ecom_rec.pipeline.recommend import Recommender
        from ecom_rec.rank.deepfm import DeepFM

        train = pl.read_parquet(PROCESSED / "train.parquet")
        user_map = pl.read_parquet(PROCESSED / "user_map.parquet")
        item_map = pl.read_parquet(PROCESSED / "item_map.parquet")

        with open(PROCESSED / "feature_spec.json") as f:
            spec = json.load(f)

        # 用户/商品统计（用于 CTR 特征）
        user_stats = train.group_by("user_id").agg([
            pl.col("rating").mean().alias("user_avg_rating"),
            pl.len().alias("user_frequency"),
            ((pl.col("timestamp_sec").max() - pl.col("timestamp_sec").min()) / 86400.0).alias("user_active_days"),
        ])
        item_stats = train.group_by("item_id").agg([
            pl.col("rating").mean().alias("item_avg_rating"),
            pl.len().alias("item_review_count"),
        ])

        # item_meta（category, brand, title）
        item_meta_cols = ["item_id"]
        for col in ["category", "brand", "title", "price"]:
            if col in train.columns:
                item_meta_cols.append(col)
        item_meta = train.select(item_meta_cols).unique(subset=["item_id"])

        # 召回模型
        pop = PopRecaller().fit(train)
        icf = ItemCFRecaller(n_neighbors=20).fit(train)
        multi_recall = MultiRecall([(pop, 0.3), (icf, 0.7)])

        # 排序模型
        dfm_path = MODELS_DIR / "deepfm.pt"
        if not dfm_path.exists():
            return None, "DeepFM 模型文件不存在（`models/deepfm.pt`），请先运行训练。"

        deepfm = DeepFM(
            dense_dim=len(spec["dense_features"]),
            sparse_vocab_sizes=spec["sparse_vocab_sizes"],
            sparse_features=spec["sparse_features"],
            embedding_dim=16,
            dnn_hidden_units=[256, 128, 64],
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        deepfm.load_state_dict(torch.load(dfm_path, map_location=device))

        rec = Recommender(
            multi_recall=multi_recall,
            rank_model=deepfm,
            dense_features=spec["dense_features"],
            sparse_features=spec["sparse_features"],
            user_stats=user_stats,
            item_stats=item_stats,
            user_map=user_map,
            item_map=item_map,
            item_meta=item_meta,
            model_type="deepfm",
            recall_k=200,
            rank_top_k=50,
            final_k=10,
            mmr_lambda=0.5,
        )
        return rec, None
    except FileNotFoundError as e:
        return None, f"数据文件缺失：{e}。请先运行 `make data`。"
    except Exception as e:
        return None, f"加载失败：{e}"

# ---- 数据是否就绪 ----
data_ready = (PROCESSED / "train.parquet").exists() and (MODELS_DIR / "deepfm.pt").exists()

if not data_ready:
    st.warning("推荐引擎尚未就绪，请先完成以下步骤：")
    st.code("""
conda activate zeolite
make data       # 下载并预处理数据
make train      # 训练所有模型（约 30-60 分钟）
    """, language="bash")
    st.info("训练完成后刷新此页面即可使用推荐演示。")

    # 演示模式：展示系统架构
    st.divider()
    st.markdown("### 推荐流水线示意（演示模式）")
    st.markdown("""
    ```
    用户 ID (input)
         │
         ▼
    ┌────────────────────────────────────┐
    │  多路召回层                         │
    │  Top-Pop ────┐                     │
    │  ItemCF ─────┼─→ 合并去重 → 200候选│
    │  BPR-MF ─────┘                     │
    └────────────────────────────────────┘
         │ 200 候选
         ▼
    ┌────────────────────────────────────┐
    │  DeepFM 精排层                      │
    │  FM 一阶 + FM 二阶交叉 + DNN        │
    │  → 按 CTR 分数排序 → Top-50         │
    └────────────────────────────────────┘
         │ Top-50
         ▼
    ┌────────────────────────────────────┐
    │  MMR 打散层 (λ=0.5)                 │
    │  相关性 vs 类目多样性 → Top-10      │
    └────────────────────────────────────┘
         │
         ▼
    个性化 Top-10 推荐结果
    ```
    """)
    st.stop()

# ---- 加载推荐引擎 ----
recommender, error_msg = load_recommender()
if recommender is None:
    st.error(error_msg)
    st.stop()

st.success("推荐引擎已就绪！")

# ---- 用户输入 ----
st.divider()
col1, col2 = st.columns([2, 1])
with col1:
    # 从训练集随机抽取用户供选择
    @st.cache_data
    def get_sample_users():
        train = pl.read_parquet(PROCESSED / "train.parquet")
        return train["user_id"].unique().sample(min(50, train["user_id"].n_unique()), seed=42).to_list()

    sample_users = get_sample_users()
    user_id = st.selectbox("选择用户 ID（或在下方输入）", [""] + sample_users)
    custom_id = st.text_input("手动输入用户 ID", placeholder="例如：A2SUAM1J3GNN3B")
    final_user_id = custom_id if custom_id else user_id

with col2:
    mmr_lambda = st.slider("MMR 多样性强度 λ", 0.0, 1.0, 0.5, 0.1,
                            help="λ=1 完全按相关性；λ=0 完全按多样性")
    recall_k = st.slider("召回候选数", 50, 300, 200, 50)

if st.button("生成个性化推荐", type="primary") and final_user_id:
    with st.spinner("正在生成推荐..."):
        recommender.mmr_lambda = mmr_lambda
        recommender.recall_k = recall_k
        result = recommender.recommend(final_user_id)

    if not result["final_top10"]:
        st.warning(f"用户 {final_user_id} 不在训练集中（冷启动用户），请换一个用户 ID。")
    else:
        st.success(f"推荐完成！用户 `{final_user_id}`")

        # ---- 三阶段可视化 ----
        tab1, tab2, tab3 = st.tabs(["多路召回（200）", "精排 Top-50", "最终推荐 Top-10"])

        with tab1:
            st.markdown(f"**多路召回候选集：{len(result['recall_candidates'])} 个商品**")
            st.caption("展示前 50 个候选")
            st.dataframe(pd.DataFrame({"商品 ID": result["recall_candidates"][:50]}),
                         use_container_width=True, height=300)

        with tab2:
            st.markdown(f"**DeepFM 精排 Top-50**")
            top50_df = pd.DataFrame({
                "排名": range(1, len(result["ranked_top50"]) + 1),
                "商品 ID": result["ranked_top50"],
                "CTR 分数": [result["scores"].get(i, 0.0) for i in result["ranked_top50"]],
            })
            st.dataframe(top50_df, use_container_width=True, height=300)

        with tab3:
            st.markdown(f"**MMR 打散后 Top-10 最终推荐**")
            final_rows = []
            for rank, item_id in enumerate(result["final_top10"], 1):
                score = result["scores"].get(item_id, 0.0)
                final_rows.append({"排名": rank, "商品 ID": item_id, "CTR 分数": f"{score:.4f}"})
            st.dataframe(pd.DataFrame(final_rows), use_container_width=True)

            # 类目分布饼图
            meta_path = PROCESSED / "train.parquet"
            if meta_path.exists() and "category" in pl.read_parquet(meta_path).columns:
                meta = pl.read_parquet(meta_path).select(["item_id", "category"]).unique()
                item_cats = {r["item_id"]: r["category"] for r in meta.iter_rows(named=True)}
                cats = [item_cats.get(i, "未知") for i in result["final_top10"]]
                fig_pie = px.pie(values=[1]*len(cats), names=cats,
                                 title="推荐结果类目分布",
                                 template="plotly_white", height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
