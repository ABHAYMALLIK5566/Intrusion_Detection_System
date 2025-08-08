# Intrusion Detection System (KDD)

Intrusion Detection using classical ML models and a Keras deep learning model trained on the KDD dataset. This repo contains reproducible notebooks and a clean project layout ready for GitHub.

## Project structure

```
.
├── artifacts/                    # Saved models and training artifacts (.pkl)
├── data/
│   └── raw/                      # Original datasets (KDDTrain.txt, KDDTest.txt)
├── docs/
│   ├── papers/                   # Project-related PDFs
│   ├── references/               # Reference PDFs
│   ├── screenshots/              # Screenshots
│   └── Final_Report.docx
├── notebooks/
│   ├── 01_training_and_saving.ipynb     # Train models and save artifacts
│   └── 02_inference_and_evaluation.ipynb# Load artifacts and evaluate/infer
├── scripts/                      # Utility scripts for formatting/refactoring
└── src/ids/                      # (optional) python package for future modularization
```

## Setup

- Python 3.10+ recommended
- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place the datasets in `data/raw/` as:
- `data/raw/KDDTrain.txt`
- `data/raw/KDDTest.txt`

The repository already includes these if available. Large raw data and generated artifacts are ignored by git.

## Usage

- Open Jupyter and run the notebooks in order:

```bash
jupyter lab  # or: jupyter notebook
```

- Start with `notebooks/01_training_and_saving.ipynb` to train and persist models/columns
- Use `notebooks/02_inference_and_evaluation.ipynb` to load artifacts and evaluate or run inference

## Deep Learning artifact

Notebook 02 expects a pre-trained DL model in `artifacts/` with one of these names:
- `model_dl.pkl` (joblib)
- `model_dl.h5` (Keras H5)
- `model_dl/` (Keras SavedModel directory)

## Notes

- Saved models are written to `artifacts/`
- If you change preprocessing (e.g., scaling, PCA), re-train and regenerate artifacts.
- Consider migrating notebook logic into `src/ids/` modules for production usage.
