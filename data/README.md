# 数据说明

本目录存放项目所用数据，原始数据不上传至 Git 仓库。

## 数据集

**Amazon Reviews 2023 — Beauty 子类目**

- 来源：https://amazon-reviews-2023.github.io/
- 规模：~70 万条评论 / ~12 万用户 / ~6 万商品
- 字段：user_id, asin(item_id), rating, timestamp, category, brand, price, title

## 下载方式

运行以下命令自动下载并预处理：

```bash
make data
```

或手动：

```bash
python scripts/00_download_data.py
python scripts/01_preprocess.py
```

## 目录结构

```
data/
├── raw/         # 原始 jsonl.gz 文件（自动下载）
├── interim/     # 清洗后中间数据（parquet 格式）
└── processed/   # 特征工程后数据，含 train/valid/test 划分
```
