"""推荐演示页面：端到端推荐链路展示（增强版）

增强内容：
- 商品信息卡片（缩略图 + 标题 + 类目 + 品牌 + CTR 分数）
- 用户历史 vs 推荐对比
- 推荐解释（来源标注）
- MMR λ 实时预览类目分布
"""
import sys
sys.path.insert(0, "src")
from pathlib import Path
from io import BytesIO
import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import json
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="推荐演示", page_icon="🛍️", layout="wide")
st.title("🛍️ 个性化推荐演示")
st.markdown(
    "输入用户 ID，查看完整推荐链路：**多路召回（200）→ DeepFM 精排（50）→ MMR 打散（10）**"
)

PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")
IMAGES_DIR = Path("app/static/images")
MAPPING_PATH = IMAGES_DIR / "mapping.json"

CATEGORY_COLORS = {
    "Skin Care": "#4CAF50",
    "Hair Care": "#2196F3",
    "Makeup": "#E91E63",
    "Bath & Body": "#FF9800",
    "Fragrance": "#9C27B0",
    "Beauty": "#00BCD4",
    "Personal Care": "#795548",
    "Tools & Accessories": "#607D8B",
    "Nail Care": "#FF5722",
    "Men's Grooming": "#3F51B5",
}

_AVATAR_CACHE = {}


def _category_color(category: str) -> str:
    category = category or ""
    for key, color in CATEGORY_COLORS.items():
        if key.lower() in category.lower():
            return color
    return "#78909C"


def _generate_avatar(item_id: str, category: str = "", size: int = 120) -> Image.Image:
    category = category or ""
    key = (item_id, category)
    if key in _AVATAR_CACHE:
        return _AVATAR_CACHE[key]

    color = _category_color(category)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    r = 16
    x0, y0, x1, y1 = 0, 0, size - 1, size - 1
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=color)

    letter = "?"
    if category:
        for word in category.split():
            if word and word[0].isalpha():
                letter = word[0].upper()
                break
    elif item_id:
        for ch in item_id:
            if ch.isalpha():
                letter = ch.upper()
                break

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size // 2)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", size // 2)
        except Exception:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (size - tw) // 2 - bbox[0]
    ty = (size - th) // 2 - bbox[1]
    draw.text((tx, ty), letter, fill="white", font=font)

    _AVATAR_CACHE[key] = img
    return img


def _show_item_image(item_id: str, category: str = "", width: int = 60):
    category = category or ""
    img_file = IMAGES_DIR / f"{item_id}.jpg"
    if img_file.exists():
        st.image(str(img_file.resolve()), width=width)
    else:
        avatar = _generate_avatar(item_id, category)
        buf = BytesIO()
        avatar.save(buf, format="PNG")
        buf.seek(0)
        st.image(buf, width=width)


@st.cache_data
def load_item_mapping():
    if MAPPING_PATH.exists():
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_item_meta():
    if not (PROCESSED / "train.parquet").exists():
        return {}
    train = pl.read_parquet(PROCESSED / "train.parquet")
    meta_cols = ["item_id"]
    for c in ["title", "category", "brand", "price"]:
        if c in train.columns:
            meta_cols.append(c)
    meta = train.select(meta_cols).unique(subset=["item_id"])
    result = {}
    for row in meta.iter_rows(named=True):
        result[row["item_id"]] = {
            "title": row.get("title") or "",
            "category": row.get("category") or "",
            "brand": row.get("brand") or "",
            "price": row.get("price", None),
        }
    return result


@st.cache_resource(show_spinner="正在加载推荐引擎（首次加载约 10-30 秒）...")
def load_recommender():
    try:
        import torch
        from ecom_rec.recall.pop import PopRecaller
        from ecom_rec.recall.itemcf import ItemCFRecaller
        from ecom_rec.pipeline.multi_recall import MultiRecall
        from ecom_rec.pipeline.recommend import Recommender
        from ecom_rec.rank.deepfm import DeepFM
        from ecom_rec.utils.device import pick_device
        import pickle

        train = pl.read_parquet(PROCESSED / "train.parquet")
        user_map = pl.read_parquet(PROCESSED / "user_map.parquet")
        item_map = pl.read_parquet(PROCESSED / "item_map.parquet")

        with open(PROCESSED / "feature_spec.json") as f:
            spec = json.load(f)

        user_stats = train.group_by("user_id").agg([
            pl.col("rating").mean().alias("user_avg_rating"),
            pl.len().alias("user_frequency"),
            ((pl.col("timestamp_sec").max() - pl.col("timestamp_sec").min()) / 86400.0).alias("user_active_days"),
        ])
        item_stats = train.group_by("item_id").agg([
            pl.col("rating").mean().alias("item_avg_rating"),
            pl.len().alias("item_review_count"),
        ])

        item_meta_cols = ["item_id"]
        for col in ["category", "brand", "title", "price"]:
            if col in train.columns:
                item_meta_cols.append(col)
        item_meta = train.select(item_meta_cols).unique(subset=["item_id"])

        pop = PopRecaller().fit(train)
        icf = ItemCFRecaller(n_neighbors=30).fit(train)
        with open(MODELS_DIR / "bpr.pkl", "rb") as f:
            bpr = pickle.load(f)
        with open(MODELS_DIR / "als.pkl", "rb") as f:
            als = pickle.load(f)

        multi_recall = MultiRecall([
            (icf, 0.35), (bpr, 0.30), (als, 0.25), (pop, 0.10),
        ])

        dfm_path = MODELS_DIR / "deepfm.pt"
        if not dfm_path.exists():
            return None, "DeepFM 模型文件不存在，请先运行训练。"

        deepfm = DeepFM(
            dense_dim=len(spec["dense_features"]),
            sparse_vocab_sizes=spec["sparse_vocab_sizes"],
            sparse_features=spec["sparse_features"],
            embedding_dim=16,
            dnn_hidden_units=[256, 128, 64],
        )
        device = pick_device()
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

        user_history = {}
        for row in train.group_by("user_id").agg(
            pl.col("item_id").alias("items"), pl.col("title").alias("titles")
        ).iter_rows(named=True):
            items = row["items"]
            titles = row["titles"]
            if hasattr(items, "to_list"):
                items = items.to_list()
                titles = titles.to_list()
            user_history[row["user_id"]] = list(zip(items, titles))

        return rec, user_history
    except FileNotFoundError as e:
        return None, f"数据文件缺失：{e}"
    except Exception as e:
        return None, f"加载失败：{e}"


def render_item_card(item_id, score=None, rank=None, source=None, img_mapping=None, meta_mapping=None):
    img_info = (img_mapping or {}).get(item_id, {})
    meta_info = (meta_mapping or {}).get(item_id, {})

    title = img_info.get("title") or meta_info.get("title") or ""
    category = img_info.get("category") or meta_info.get("category") or ""
    brand = img_info.get("brand") or meta_info.get("brand") or ""
    price = img_info.get("price") or meta_info.get("price")

    if not title:
        parts = []
        if brand:
            parts.append(brand)
        if category:
            parts.append(category.split(",")[0].split("&")[0].strip())
        title = " ".join(parts) if parts else f"商品 {item_id[:6]}"

    cols = st.columns([1, 4])
    with cols[0]:
        _show_item_image(item_id, category, width=60)

    with cols[1]:
        header = f"**{title[:60]}**"
        if rank:
            header = f"#{rank} {header}"
        st.markdown(header)
        details = []
        if category:
            details.append(f"📂 {category[:30]}")
        if brand:
            details.append(f"🏷️ {brand[:20]}")
        if price and isinstance(price, (int, float)):
            details.append(f"💰 ${price:.2f}")
        if score is not None:
            details.append(f"📊 CTR={score:.4f}")
        if source:
            details.append(f"🔗 {source}")
        if details:
            st.caption(" | ".join(details))


data_ready = (PROCESSED / "train.parquet").exists() and (MODELS_DIR / "deepfm.pt").exists()

if not data_ready:
    st.warning("推荐引擎尚未就绪，请先完成以下步骤：")
    st.code("conda run -n data python scripts/02_train_recall.py\nconda run -n data python scripts/03_train_rank.py", language="bash")
    st.stop()

recommender, user_history_or_error = load_recommender()
if recommender is None:
    st.error(str(user_history_or_error))
    st.stop()

user_history = user_history_or_error if isinstance(user_history_or_error, dict) else {}
img_mapping = load_item_mapping()
meta_mapping = load_item_meta()

st.success("推荐引擎已就绪！")

st.divider()
col1, col2 = st.columns([2, 1])
with col1:
    @st.cache_data
    def get_sample_users():
        train = pl.read_parquet(PROCESSED / "train.parquet")
        active = train.group_by("user_id").len().sort("len", descending=True)
        top_users = active.head(100)["user_id"].to_list()
        return top_users[:50]

    sample_users = get_sample_users()
    user_id = st.selectbox("选择活跃用户 ID（或在下方输入）", [""] + sample_users)
    custom_id = st.text_input("手动输入用户 ID", placeholder="例如：A2SUAM1J3GNN3B")
    final_user_id = custom_id if custom_id else user_id

with col2:
    mmr_lambda = st.slider("MMR 多样性强度 λ", 0.0, 1.0, 0.5, 0.05,
                            help="λ=1 完全按相关性；λ=0 完全按多样性")
    recall_k = st.slider("召回候选数", 50, 300, 200, 50)

if st.button("🚀 生成个性化推荐", type="primary") and final_user_id:
    with st.spinner("正在生成推荐..."):
        recommender.mmr_lambda = mmr_lambda
        recommender.recall_k = recall_k
        result = recommender.recommend(final_user_id)

    if not result["final_top10"]:
        st.warning(f"用户 {final_user_id} 不在训练集中（冷启动用户），请换一个用户 ID。")
    else:
        st.success(f"推荐完成！用户 `{final_user_id}`")

        # ---- 用户历史 vs 推荐对比 ----
        history_items = user_history.get(final_user_id, [])
        if history_items:
            st.markdown("### 👤 用户购买历史（最近 10 条）")
            hist_cols = st.columns(min(5, len(history_items[:10])))
            for i, (iid, ititle) in enumerate(history_items[:10]):
                with hist_cols[i % 5]:
                    m = meta_mapping.get(iid, {})
                    cat = m.get("category") or ""
                    _show_item_image(iid, cat, width=60)
                    display_title = ititle if ititle else ""
                    if not display_title:
                        m2 = meta_mapping.get(iid, {})
                        parts = []
                        if m2.get("brand"):
                            parts.append(m2["brand"])
                        if m2.get("category"):
                            parts.append(m2["category"].split(",")[0].split("&")[0].strip())
                        display_title = " ".join(parts) if parts else f"商品 {iid[:6]}"
                    st.caption(f"{display_title[:30]}")

        st.divider()

        # ---- 最终推荐 Top-10 卡片 ----
        st.markdown("### 🎯 个性化推荐 Top-10")
        st.markdown(f"MMR λ={mmr_lambda:.2f}（{'偏相关性' if mmr_lambda > 0.6 else '偏多样性' if mmr_lambda < 0.4 else '均衡'}）")

        for rank, item_id in enumerate(result["final_top10"], 1):
            score = result["scores"].get(item_id, 0.0)
            render_item_card(
                item_id, score=score, rank=rank,
                source="精排+MMR", img_mapping=img_mapping, meta_mapping=meta_mapping,
            )

        # ---- 类目分布饼图 ----
        st.divider()
        col_pie1, col_pie2 = st.columns(2)

        with col_pie1:
            top50_cats = []
            for iid in result["ranked_top50"][:50]:
                m = meta_mapping.get(iid, {})
                cat = m.get("category") or "未知"
                top50_cats.append(cat if cat else "未知")
            fig1 = px.pie(values=[1]*len(top50_cats), names=top50_cats,
                          title=f"精排 Top-50 类目分布",
                          template="plotly_white", height=300)
            st.plotly_chart(fig1, use_container_width=True)

        with col_pie2:
            final_cats = []
            for iid in result["final_top10"]:
                m = meta_mapping.get(iid, {})
                cat = m.get("category") or "未知"
                final_cats.append(cat if cat else "未知")
            fig2 = px.pie(values=[1]*len(final_cats), names=final_cats,
                          title=f"MMR Top-10 类目分布（λ={mmr_lambda:.2f}）",
                          template="plotly_white", height=300)
            st.plotly_chart(fig2, use_container_width=True)

        # ---- 三阶段详情（可展开） ----
        with st.expander("📊 查看推荐链路详情"):
            tab1, tab2, tab3 = st.tabs(["多路召回（200）", "精排 Top-50", "MMR Top-10"])

            with tab1:
                st.markdown(f"**多路召回候选集：{len(result['recall_candidates'])} 个商品**")
                st.dataframe(pd.DataFrame({"商品 ID": result["recall_candidates"][:50]}),
                             use_container_width=True, height=300)

            with tab2:
                top50_df = pd.DataFrame({
                    "排名": range(1, len(result["ranked_top50"]) + 1),
                    "商品 ID": result["ranked_top50"],
                    "CTR 分数": [result["scores"].get(i, 0.0) for i in result["ranked_top50"]],
                })
                st.dataframe(top50_df, use_container_width=True, height=300)

            with tab3:
                final_rows = []
                for rank, item_id in enumerate(result["final_top10"], 1):
                    score = result["scores"].get(item_id, 0.0)
                    m = meta_mapping.get(item_id, {})
                    final_rows.append({
                        "排名": rank, "商品 ID": item_id,
                        "类目": m.get("category") or "",
                        "CTR 分数": f"{score:.4f}",
                    })
                st.dataframe(pd.DataFrame(final_rows), use_container_width=True)
