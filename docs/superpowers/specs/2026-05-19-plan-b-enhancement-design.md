# 方案 B 设计文档：Bug 修复 + 模型调优 + Demo 增强

**日期**: 2026-05-19
**项目**: 电商用户购买行为分析与个性化推荐
**目标**: 修复致命 bug，调优模型指标，增强 Streamlit Demo 展示效果

---

## 一、Bug 修复（3 个）

### Bug 1: BPR-MF 几乎失效（HR@10=0.0008）

**文件**: `src/ecom_rec/recall/bpr.py:57`

**根因**: 使用原始评分 (1-5) 作为 user_item 矩阵置信度。implicit BPR 按置信度加权采样正样本，评分 5 的采样概率是评分 1 的 5 倍，导致训练分布偏斜、模型退化。

**修复**: `data = interactions["rating"].to_list()` → `data = [1.0] * len(rows)`

**配套**: 删除旧 `models/bpr.pkl`，强制重训。

**预期**: BPR HR@50 从 0.18% 提升到 5-12%。

### Bug 2: GAUC == AUC（值完全相同）

**文件**: `scripts/03_train_rank.py:95`

**根因**: `_evaluate_on_test` 传入 `test_df["user_idx"]`，负采样 1:4 后每用户内部正负比例高度一致，导致 GAUC ≈ AUC。需验证是否为上次 fast 模式训练的结果。

**修复**:
1. 确保全量训练（sample_ratio=1.0）后重新评估
2. 在 `_evaluate_on_test` 中添加日志：打印用户级 AUC 的 mean/std/min/max，确认 GAUC 计算有意义
3. 如果全量训练后 GAUC 仍 == AUC，说明 1:4 负采样结构确实导致用户内分布趋同，需在报告中解释

### Bug 3: DeepFM 仅训练 2 轮 early stop

**文件**: `configs/rank/deepfm.yaml`, `configs/rank/widedeep.yaml`

**根因**: 可能上次在 fast 模式下训练（sample_ratio=0.1），验证集 AUC 第 2 轮就不再提升。

**修复**:
1. 确保 `make train` 使用 sample_ratio=1.0
2. `lr: 0.001` → `0.0005`
3. `early_stopping_patience: 3` → `5`
4. 同步修改 widedeep.yaml

---

## 二、模型调优

### 2.1 召回层

| 模型 | 改动 | 训练时间(M5) |
|------|------|-------------|
| BPR-MF | 只修 confidence，参数不变（factors=64, iterations=100, lr=0.01） | ~20 min |
| ALS | factors=64→128, 其余不变 | ~15 min |
| ItemCF | n_neighbors=20→50 | ~10 min |
| Top-Pop | 不变 | <1 min |

**策略**: 先修 BPR bug 跑第 1 轮，看结果再决定是否微调。目标 MultiRecall HR@50 > 10%。

### 2.2 精排层

| 模型 | 改动 | 训练时间(M5) |
|------|------|-------------|
| DeepFM | lr=5e-4, patience=5, sample_ratio=1.0 | ~60-90 min |
| Wide&Deep | 同上 | ~60-90 min |
| LightGBM | n_estimators=500, num_leaves=63 | ~5-8 min |

**内存安全**: 16GB 统一内存，CTR 训练集 ~210 万条 × batch_size=4096，峰值约 6-8GB，安全。

### 2.3 端到端评估

**新增**: 在 `scripts/03_train_rank.py` 末尾或独立脚本中，采样 3000 测试用户跑 `Recommender.recommend()`，计算 Top-10 的 HR@10 / NDCG@10 / Diversity。输出到 `reports/e2e_benchmark.json`。

**耗时**: ~30 min（3000 用户 × 0.5s/用户）。

---

## 三、Demo 增强

### 3.1 预下载商品图片

**新增脚本**: `scripts/download_images.py`

- 从 `data/raw/meta_Beauty_and_Personal_Care.jsonl.gz` 提取 Top-500 热门商品的 `thumb` 图片 URL
- 多线程下载到 `app/static/images/{item_id}.jpg`
- 总大小约 1-2MB
- 降级：图片不存在时 fallback 到文本

### 3.2 推荐演示页重写

**改动文件**: `app/pages/4_🛍️_推荐演示.py`

增强内容：
1. **商品卡片**: 缩略图 + 标题 + 类目 + 品牌 + 价格 + CTR 分数
2. **用户历史 vs 推荐对比**: 左栏用户购买历史，右栏推荐结果
3. **推荐解释**: 每个推荐标注来源（"同类目推荐"/"协同过滤"/"热门兜底"）
4. **MMR 实时预览**: 拖动 λ 滑块后立即显示类目分布变化图

### 3.3 首页增强

**改动文件**: `app/Home.py`

- 系统架构流程图（Mermaid 渲染）
- 关键数字概览卡片（用户数、商品数、AUC、HR@50）
- "一键体验"按钮跳转到推荐演示页

### 3.4 新增端到端指标页

**新增文件**: `app/pages/5_📈_端到端评估.py`

- 展示 Top-10 推荐的 HR@10 / NDCG@10 / Diversity
- 与中间层指标（召回 HR@50、精排 AUC）的对比表
- 读取 `reports/e2e_benchmark.json`

### 3.5 内存约束

所有增强均为轻量操作：图片预加载 ~2MB，item_meta ~200MB，JSON <1MB。16GB 安全。

---

## 四、报告一致性

修复 bug + 重训后，用 `report_writer.py` 统一回填 `README.md`、`reports/03_推荐模型对比报告.md` 中的 AUTO 标记块。

---

## 五、实施时间线

| 阶段 | 任务 | 耗时 | 依赖 |
|------|------|------|------|
| **Week 1** | | | |
| Day 1-2 | 修复 BPR confidence + 删除旧 pkl | 2h | 无 |
| Day 2 | 修复 DeepFM/W&D patience/lr 配置 | 1h | 无 |
| Day 3 | 重跑召回训练 → 检查 BPR 指标 | 1h 等 | BPR fix |
| Day 3-4 | 微调参数 → 重跑召回 | ~1h 等 | 第1轮 |
| Day 4-5 | 重跑精排全量训练 | ~2-3h 等 | 召回完成 |
| Day 5 | 端到端评估脚本 + 执行 | 2h + 30min | 精排完成 |
| **Week 2** | | | |
| Day 6 | 下载 Top-500 图片脚本 + 执行 | 2h + 10min | 无 |
| Day 7-8 | 推荐演示页重写 | 1天 | 图片就绪 |
| Day 9 | 首页 + 端到端指标页 | 半天 | 评估完成 |
| Day 10 | 回填报告/README + 全链路测试 | 半天 | 全部 |
| Day 11-12 | 缓冲：修 Demo bug、演练 | — | — |
