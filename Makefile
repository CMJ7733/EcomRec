.PHONY: install data test train train-fast figures app lint clean all

PYTHON := python
PIP := pip
NB_EXEC := jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3

install:
	$(PIP) install -e ".[dev]"

data:
	$(PYTHON) scripts/00_download_data.py
	$(PYTHON) scripts/01_preprocess.py

train:
	$(PYTHON) scripts/02_train_recall.py
	$(PYTHON) scripts/03_train_rank.py

# M 芯片本地快速验证：缩减 epoch/iter，~25 分钟跑完
train-fast:
	$(PYTHON) scripts/02_train_recall.py fast=true
	$(PYTHON) scripts/03_train_rank.py fast=true

# 执行 notebooks 生成 reports/figures/*.png（依赖 data + train 产物）
figures:
	$(NB_EXEC) notebooks/01_eda.ipynb
	$(NB_EXEC) notebooks/02_rfm_user_profile.ipynb
	$(NB_EXEC) notebooks/03_recall_benchmark.ipynb
	$(NB_EXEC) notebooks/04_rank_benchmark.ipynb
	$(NB_EXEC) notebooks/05_end2end_case_study.ipynb

# 一键全量：data → train → figures
all: data train figures

test:
	pytest -v tests/

app:
	streamlit run app/Home.py

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .ruff_cache/ mlruns/
