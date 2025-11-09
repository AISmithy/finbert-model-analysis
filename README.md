# FinBERT Model Analysis — Baseline Evaluation

This repository contains a small pipeline to evaluate a FinBERT-based baseline classifier on financial headlines. The README summarizes the implementation, how to run the scripts, and an analysis report from the latest execution (artifacts live under `runs/baseline_finbert`).

## Implementation (high-level)

- `src/sentiment_analyzer.py`
  - Provides `load_sentiment_model()` which loads the `ProsusAI/finbert` model via Hugging Face `transformers` pipeline.
  - Provides `analyze_sentiment(texts, model=None)` which is robust to different calling orders and accepts a single string or list of strings. It returns a prediction dict for single inputs or a list of prediction dicts for lists.
  

- `src/data_ingestion.py`
  - Contains helper(s) to fetch news for a given ticker. Used by the dataset builder script.

- `scripts/build_silver_dataset.py`
  - Fetches recent news for a ticker, labels each headline with the FinBERT pipeline, and writes a lightweight CSV suitable for training/evaluation (`data/headlines.csv` by default).
  - Usage example: `python scripts/build_silver_dataset.py --ticker AAPL --out data/headlines.csv --limit 300 --unique`.

- `src/baseline_finbert_eval.py`
  - Loads a labeled CSV, cleans text rows, splits into train/test, runs FinBERT to predict sentiment on the test set, computes metrics, and writes artifacts to an output folder.
  - Artifacts written to `<out>` (default `runs/baseline_finbert`):
    - `classification_report.txt` — sklearn classification report
    - `confusion_matrix.png` — confusion matrix heatmap (PNG)
    - `predictions.csv` — test examples with predicted labels
    - `metrics.json` — summary metrics and run metadata

## Requirements / Environment

All third-party Python dependencies are listed in `requirements.txt`. Key packages include:

- numpy, pandas, matplotlib
- scikit-learn
- transformers (Hugging Face)
- torch (PyTorch backend for transformers)
- yfinance, yahoo_fin (data ingestion helpers)
- streamlit (optional; used conditionally)

On Windows (PowerShell) with the provided virtual environment, use:

```powershell
# If you haven't created a venv yet
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the repository already configured a virtual env at `.venv`, use that Python executable to run scripts:

```powershell
C:/GIT-REPO/finbert-model-analysis/.venv/Scripts/python.exe scripts/build_silver_dataset.py --ticker AAPL --out data/headlines.csv --limit 300 --unique

C:/GIT-REPO/finbert-model-analysis/.venv/Scripts/python.exe src/baseline_finbert_eval.py --data data/headlines.csv --text-col headline --label-col label --test-size 0.2 --out runs/baseline_finbert
```

## Execution procedure (quick)

1. Prepare environment and install dependencies (see above).
2. Build a labeled (silver) dataset (optionally):
   - Example: `python scripts/build_silver_dataset.py --ticker AAPL --out data/headlines.csv --limit 300 --unique`
   - This script fetches recent news, labels headlines with the FinBERT pipeline, and writes `data/headlines.csv`.
3. Run the baseline evaluation:
   - `python src/baseline_finbert_eval.py --data data/headlines.csv --text-col headline --label-col label --test-size 0.2 --out runs/baseline_finbert`
4. Inspect artifacts in `runs/baseline_finbert` (classification report, confusion matrix PNG, predictions CSV, and metrics JSON).

## Latest run — analysis report (from `runs/baseline_finbert`)

Files examined for this report:
- `runs/baseline_finbert/metrics.json`
- `runs/baseline_finbert/classification_report.txt`
- `runs/baseline_finbert/predictions.csv`
- `runs/baseline_finbert/confusion_matrix.png`

Key metadata (extracted from `metrics.json`):

- Timestamp: 2025-11-08T17:22:26.718402Z
- Model used: `ProsusAI/finbert` (Hugging Face pipeline)
- Test size: 0.2 (20% held out)
- Number of training samples: 16
- Number of test samples: 4
- Classes observed: ["negative", "neutral"]
- Accuracy: 1.0
- Macro F1: 1.0

Classification report (from `classification_report.txt`):

```
              precision    recall  f1-score   support

    negative       1.00      1.00      1.00         1
     neutral       1.00      1.00      1.00         3

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```

Observations and interpretation

- The baseline produced perfect scores on the held-out test set (accuracy = 1.0 and macro-F1 = 1.0). However, the absolute test set is very small (n_test = 4). Metrics computed on such a small test set are unreliable and susceptible to high variance.

- Class distribution in the run was imbalanced: only two classes were present (`negative` and `neutral`) with supports 1 and 3 respectively. No `positive` samples were present in that particular split.

- High-level recommendation: evaluate on a larger and more representative test set. Consider k-fold cross-validation or repeated stratified splits to get stable performance estimates.

Caveats / Limitations

- The script relies on the Hugging Face transformers pipeline and a PyTorch backend (`torch`). If CUDA/GPU is not available, models run on CPU and can be slow for large datasets.

- The dataset builder (`scripts/build_silver_dataset.py`) uses programmatic labeling via FinBERT. Those labels are "silver" (no human validation). For a robust supervised baseline you should collect human-labeled gold data or perform spot-checks of the silver labels.

- The `analyze_sentiment` function is tolerant to different argument orders for backward compatibility. Callers should use the canonical `(texts, model)` ordering where `model` is the pipeline returned by `load_sentiment_model()`.


## Where to find artifacts

- Data: `data/headlines.csv`
- Run outputs: `runs/baseline_finbert/` (classification_report.txt, confusion_matrix.png, predictions.csv, metrics.json)
- Core code: `src/` (data_ingestion.py, sentiment_analyzer.py, baseline_finbert_eval.py)


