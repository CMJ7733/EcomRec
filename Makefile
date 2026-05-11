.PHONY: install data test train app lint clean

PYTHON := python
PIP := pip

install:
	$(PIP) install -e ".[dev]"

data:
	$(PYTHON) scripts/00_download_data.py
	$(PYTHON) scripts/01_preprocess.py

train:
	$(PYTHON) scripts/02_train_recall.py
	$(PYTHON) scripts/03_train_rank.py

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