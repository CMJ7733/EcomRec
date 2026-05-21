# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

This is a Chinese-language data-mining course assignment ("数据挖掘大作业") implementing an industrial-style two-stage recommendation system on Amazon Reviews 2023 (Beauty subset). It is a **single-developer project**, not a production service — be pragmatic, avoid speculative abstractions.

## Environment

- **Always run inside the `data` conda env.** The default `python` (miniforge base) lacks torch/polars/lightgbm and will fail with `ModuleNotFoundError`. Either `conda activate data` once per shell, or prefix one-off commands: `conda run -n data python ...`.
- **Apple Silicon (M-chip) Mac**: there's no CUDA. The codebase uses `pick_device()` to select MPS when available.
- **LightGBM requires libomp on macOS** — if `import lightgbm` fails with `Library not loaded: libomp.dylib`, run `brew install libomp`.
- **LightGBM + torch coexistence segfault** (M-chip): torch and sklearn each bundle their own `libomp.dylib`. Loading them in the same process as LightGBM causes `SIGSEGV` mid-training. **Fix: `import lightgbm` BEFORE any `import torch` / `import sklearn`.** See the preamble of [scripts/03_train_rank.py](scripts/03_train_rank.py) for the pattern. `KMP_DUPLICATE_LIB_OK=TRUE` alone is **not** sufficient on Apple Silicon — the import-order trick is what actually works.

## Common commands

`make` targets are thin wrappers around `python scripts/...`. The Makefile is at the repo root.

```bash
make data         # 00_download_data.py + 01_preprocess.py — ~5 min
make train        # 02_train_recall.py + 03_train_rank.py (full hyperparams) — ~1-2h on M-chip
make train-fast   # same with fast=true override — ~25 min, for local iteration
make figures      # nbconvert --execute on notebooks/01-05 → reports/figures/*.png
make all          # data + train + figures
make test         # pytest -v tests/
make app          # streamlit run app/Home.py
make lint         # ruff check + format --check
```

To run a **single test**: `pytest -v tests/test_pipeline.py::test_mmr_diversity`.

To run a training script directly with config override: `python scripts/02_train_recall.py fast=true` (OmegaConf CLI syntax, not argparse). Anything in `configs/config.yaml` or merged sub-configs can be overridden this way.

## Architecture overview

The pipeline is **data → features → recall (4 models) → CTR features → rank (3 models) → MMR rerank**. Each stage is a separate module in `src/ecom_rec/`. The training scripts and the Streamlit app are both thin orchestrators on top of these modules.

```
data.clean.py        →  K-core(5) filter + time feature extraction → data/interim/interactions.parquet
data.split.py        →  time-based 80/10/10 + LOO valid → data/processed/{train,valid,test}.parquet + user_map/item_map
features.rfm.py      →  R/F/M aggregation per user (M ≈ rating × price with category-mean imputation)
features.profile.py  →  StandardScaler + KMeans(k=4) → labels 4 segments by F+M-R ranking
features.ctr_features.py → DENSE(6) + SPARSE(6) features, neg-sample 1:4 (train/valid/test all) → ctr_*.parquet + feature_spec.json
recall/{pop,itemcf,bpr,als}.py → all implement Recaller ABC (fit, recommend, recommend_batch, name)
rank/{lgb,deepfm,widedeep}.py + rank/trainer.py → CTR scorers; trainer.py is the generic torch training loop with AMP + tqdm + early stopping
pipeline/multi_recall.py → weighted position-decay fusion of multiple Recallers (default weights from ADR-4)
pipeline/rerank.py    → MMR (λ-balanced relevance vs category-diversity)
pipeline/recommend.py → end-to-end Recommender class: multi_recall(200) → rank(50) → MMR(10)
```

### Non-obvious design choices (see [docs/design_decisions.md](docs/design_decisions.md) for the full ADRs)

- **No data leakage**: `build_ctr_features` computes all statistics (user_avg_rating, item_review_count, price quantiles) **only from train**, never from valid/test. Time-based splitting + LOO valid is mandatory.
- **Negative sampling is in 3 splits, not just train**: a recent bug-fix in `build_ctr_features` extended neg sampling to valid/test, otherwise val_AUC=0 and early stopping fails. Don't revert.
- **`implicit` library quirks**: BPR and ALS `recommend()` requires a CSR slice (`user_item_matrix[uid]`), not `None`. Both Recallers also apply a defensive post-hoc history filter — `filter_already_liked_items=True` is unreliable on small/sparse user histories.
- **Device selection is centralized** in `src/ecom_rec/utils/device.py::pick_device()` (CUDA > MPS > CPU). When adding model code that uses torch, **do not** write `torch.device("cuda" if ... else "cpu")` — that misses MPS on M-chip Macs.
- **AMP only on CUDA**: trainer.py guards `GradScaler` / `autocast` with `device.type == "cuda"`. MPS runs fp32.
- **Reports use AUTO markers**: `reports/03_推荐模型对比报告.md`, `README.md`, and `reports/02_用户画像报告.md` contain `<!-- AUTO:xxx -->...<!-- /AUTO -->` blocks that training scripts (via `src/ecom_rec/utils/report_writer.py`) rewrite. Don't manually edit content between AUTO markers — it'll be overwritten next training run. Sections outside markers are safe to edit.
- **Polars-first**: `polars` is the canonical DataFrame; `pandas` only at sklearn/streamlit boundaries. When adding code that joins/aggregates data, prefer polars expressions.

### Config system

- Configs are YAML loaded via **OmegaConf**, not Hydra (despite the `defaults:` block in `configs/config.yaml` which is leftover and unused).
- `_load_cfg()` in each training script merges `configs/config.yaml` + sub-configs + CLI overrides.
- `fast=true` is a top-level toggle that scripts use to override per-model hyperparams (epochs, iterations, n_estimators) in-memory — the YAML files are not touched.

## Notebooks vs scripts

- `scripts/` is the **canonical training entrypoint** (callable by `make train`, reproducible, used by CI if added).
- `notebooks/` are for **visualization and analysis**, not training logic. They reuse `src/ecom_rec/` modules and produce figures into `reports/figures/`. Notebook 04 will re-train DeepFM if `models/deepfm.pt` doesn't exist — ensure scripts have run first.
- Old `scripts/verify_*.py` are smoke tests for module interfaces; keep them passing when modifying corresponding modules (e.g. `verify_recall.py` asserts recommender history filtering).

### Notebook path / CWD gotcha (VSCode)

VSCode Jupyter kernel sets CWD to the **repo root**, not the `notebooks/` directory. Do **not** use `sys.path.insert(0, "../src")` in notebooks — it resolves to the repo's parent and `ecom_rec` won't be found. The correct pattern (already in notebook cell-1) auto-detects CWD:

```python
_nb_dir = Path(__file__).parent if "__file__" in dir() else Path.cwd()
for _p in [_nb_dir / "src", _nb_dir.parent / "src"]:
    if (_p / "ecom_rec").exists():
        sys.path.insert(0, str(_p))
        break
```

Similarly, `DATA_DIR` and `FIGURES` must be resolved from CWD, not hardcoded relative paths.

### Notebook output dependency: `rfm_labeled.parquet`

`notebooks/02_rfm_user_profile.ipynb` must be run to completion to generate `data/processed/rfm_labeled.parquet`. The Streamlit 用户画像 page reads this file; without it the page shows a warning and stops. The save cell is already present in the notebook (after the clustering cell).

### kaleido on Apple Silicon

`plotly`'s `fig.write_image()` requires kaleido. **kaleido ≥ 1.x is broken on M-chip Macs** — install the pinned version:

```bash
conda run -n data pip install "kaleido==0.2.1"
```

`requirements.txt` pins this version. Do not upgrade kaleido without testing on Apple Silicon first.

## Code style

- All docstrings, log messages, and inline comments are in Chinese — follow this convention when extending code.
- Each training script starts with `sys.path.insert(0, "src")` and **must be run from the repo root** (the conftest/path setup assumes CWD = repo root).
- `from __future__ import annotations` at the top of most modules — keep it.
- Logging uses `ecom_rec.utils.logger.get_logger` (Rich-formatted). Don't use `print()` in library code.

## Data state

- `data/raw/` (Beauty jsonl.gz, ~1GB), `data/interim/`, `data/processed/` are all gitignored.
- `models/` is gitignored — model artifacts are reproducible from `make train`.
- On a fresh clone, run `make data && make train-fast` to materialize everything needed for `make app` to work.
