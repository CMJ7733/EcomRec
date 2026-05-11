<div align="center">

# 🛍️ EcomRec — 电商用户行为分析与深度混合推荐系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Amazon%20Reviews%202023-orange)](https://amazon-reviews-2023.github.io/)

基于 Amazon Reviews 2023 数据集，完整复现工业界"召回-精排"双塔推荐架构。

[快速开始](#-快速开始) • [系统架构](#️-系统架构) • [实验结果](#-实验结果) • [项目结构](#-项目结构) • [分析报告](#-分析报告)

</div>

---

## ✨ 核心特性

- **多维用户画像**：基于 RFM 模型的 K-Means 聚类，识别高价值/潜力/沉睡/流失四类用户群体
- **多路召回层**：Top-Pop / ItemCF / BPR-MF / ALS 四路召回并行，候选集融合去重
- **深度精排层**：LightGBM（Baseline）→ DeepFM（FM + DNN）→ Wide & Deep 逐步提升
- **全链路推荐**：多路召回（200）→ DeepFM 精排（50）→ MMR 打散（10），防止信息茧房
- **交互式 Dashboard**：Streamlit 4 页面展示用户画像、召回对比、排序分析、实时推荐演示

## 🏗️ 系统架构

```
原始数据 (Amazon Reviews 2023 - Beauty)
    │
    ▼
数据治理层
├── K-core(5) 过滤 → 去除低活跃用户/商品
├── 时间戳解析 → 提取周期性特征
└── 时间序切分 → Train(80%) / Valid(10%) / Test(10%)
    │
    ▼
用户画像层
├── RFM 建模 (Recency / Frequency / Monetary)
└── KMeans 聚类 → 4 类用户群体画像
    │
    ▼
召回层 (Retrieval)
├── Top-Pop     → 全局热门兜底
├── ItemCF      → 共现矩阵 + IUF 衰减
├── BPR-MF      → 贝叶斯个性化排序
└── ALS         → 交替最小二乘矩阵分解
    │  多路融合 → 200 候选
    ▼
精排层 (Ranking)
├── LightGBM    → Pointwise/Pairwise Baseline
├── DeepFM      → FM 低阶交叉 + DNN 高阶特征 (PyTorch 原生)
└── Wide & Deep → Wide(LR) + Deep(DNN)
    │  Top-50
    ▼
后处理层
└── MMR 打散    → λ=0.5 类目多样性保证 → Top-10
    │
    ▼
Streamlit Dashboard
```

## 🚀 快速开始

### 环境要求

- Python 3.10 / 3.11
- CUDA 12.1+（可选，CPU 也可运行）
- 磁盘空间：约 5GB（数据 + 模型）

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/EcomRec.git
cd EcomRec

# 安装依赖
make install
```

### 数据准备

```bash
make data
# 自动下载 Amazon Reviews 2023 Beauty 子集并完成清洗
# 产出：data/processed/train.parquet / valid.parquet / test.parquet
```

### 训练所有模型

```bash
make train
# 训练时间：~30-60 分钟（GPU）/ ~2-4 小时（CPU）
# MLflow 追踪：mlflow ui
```

### 启动 Dashboard

```bash
make app
# 浏览器访问 http://localhost:8501
```

## 📊 实验结果

> 结果将在模型训练完成后更新

### 召回层对比（Amazon Beauty，K-core=5）

| 模型 | HR@10 | HR@50 | Recall@50 | NDCG@50 |
|------|-------|-------|-----------|---------|
| Top-Pop | - | - | - | - |
| ItemCF | - | - | - | - |
| BPR-MF | - | - | - | - |
| ALS | - | - | - | - |

### 精排层对比

| 模型 | AUC | LogLoss | GAUC |
|------|-----|---------|------|
| LightGBM | - | - | - |
| Wide & Deep | - | - | - |
| **DeepFM** | - | - | - |

## 📁 项目结构

```
EcomRec/
├── configs/          # Hydra 配置文件
├── data/             # 数据目录（见 data/README.md）
├── notebooks/        # 叙事化分析 Notebook
├── src/ecom_rec/     # 核心源码包
│   ├── data/         # 数据下载、清洗、切分
│   ├── features/     # RFM、CTR 特征工程
│   ├── recall/       # 召回模型
│   ├── rank/         # 排序模型
│   ├── eval/         # 评估指标
│   ├── pipeline/     # 推荐流水线
│   └── utils/        # 工具函数
├── app/              # Streamlit Dashboard
├── tests/            # 单元测试
├── reports/          # 分析报告与图表
├── scripts/          # 训练脚本
└── docs/             # 架构文档
```

## 📝 分析报告

- [📊 01 — 数据探索分析（EDA）](reports/01_数据分析报告.md)
- [👥 02 — 用户画像报告](reports/02_用户画像报告.md)
- [🔍 03 — 推荐模型对比报告](reports/03_推荐模型对比报告.md)
- [📋 04 — 项目总结与经验沉淀](reports/04_项目总结.md)

## 🛠️ 技术栈

| 模块 | 技术 |
|------|------|
| 数据处理 | Polars, PyArrow, Pandas |
| 经典推荐 | Scikit-learn, implicit |
| 深度学习 | PyTorch 2.4, torchmetrics |
| 实验追踪 | MLflow |
| 配置管理 | Hydra-core, OmegaConf |
| 可视化 | Plotly, Matplotlib, Seaborn |
| 前端展示 | Streamlit |

## 🤝 参考资料

- [Amazon Reviews 2023 数据集](https://amazon-reviews-2023.github.io/)
- DeepFM: Guo et al., *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*, IJCAI 2017
- BPR: Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009
- Wide & Deep: Cheng et al., *Wide & Deep Learning for Recommender Systems*, RecSys 2016

---

<div align="center">
<sub>以资深数据挖掘工程师视角复现工业级推荐架构 · 数据挖掘课程大作业</sub>
</div>
