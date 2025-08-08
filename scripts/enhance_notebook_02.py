import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / 'notebooks' / '02_inference_and_evaluation.ipynb'

TITLE_MD = (
    "# Intrusion Detection — Inference and Evaluation\n\n"
    "This notebook loads pre-trained models and evaluates them on the KDD test data.\n"
    "It mirrors the structure of the training notebook and is designed to be run top-to-bottom.\n"
)

HEADINGS = {
    'imports': '## Setup and Imports',
    'data': '## Load Test Data',
    'preprocess': '## Preprocessing',
    'scaling': '### Scaling (RobustScaler)',
    'pca': '### Dimensionality Reduction (PCA)',
    'split': '## Train/Test Split for Evaluation',
    'load_models': '## Load Trained Models (Classical ML)',
    'metrics': '## Metrics and Evaluation Helpers',
    'eval_ml': '## Evaluate Classical ML Models',
    'dl': '## Deep Learning (Loaded Model)',
    'end': '## Next Steps',
}

MARKERS = {
    'imports': ['import pandas as pd', 'import numpy as np'],
    'data': ['pd.read_csv("../data/raw/KDDTest.txt")', "pd.read_csv('../data/raw/KDDTest.txt')"],
    'preprocess': ['Manual mapping for protocol_type', 'protocol_map ='],
    'scaling': ['RobustScaler()'],
    'pca': ['PCA(n_components', '## PCA'],
    'split': ['train_test_split(x, y', 'x_train, x_test, y_train, y_test = train_test_split'],
    'load_models': ["joblib.load('../artifacts/model_lr.pkl')"],
    'metrics': ['def evaluate_classification('],
    'eval_ml': ['evaluate_classification(model_lr', 'evaluate_classification('],
    'dl': ['# Load pre-trained Deep Learning model from artifacts'],
}

END_MD = (
    "- Ensure the DL artifact exists at `../artifacts/model_dl.pkl` (or SavedModel/H5).\n"
    "- Re-run from the top to reproduce results and refresh plots.\n"
)

def insert_heading_before(cells, idx, text):
    cells.insert(idx, {'cell_type': 'markdown', 'metadata': {}, 'source': text})


def enhance():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])

    # Ensure title at top
    if not cells or cells[0].get('cell_type') != 'markdown' or '# Intrusion Detection' not in ''.join(cells[0].get('source', '')):
        cells.insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': TITLE_MD})

    # Scan and insert headings before first occurrence of each marker category
    inserted = {k: False for k in HEADINGS}

    i = 0
    while i < len(cells):
        cell = cells[i]
        src = ''.join(cell.get('source', [])) if isinstance(cell.get('source'), list) else cell.get('source', '')
        if cell.get('cell_type') == 'code':
            for key, patterns in MARKERS.items():
                if not inserted.get(key):
                    if any(pat in src for pat in patterns):
                        insert_heading_before(cells, i, HEADINGS[key] + '\n')
                        inserted[key] = True
                        i += 1  # Skip over inserted heading
                        break
        i += 1

    # Append end notes
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': HEADINGS['end'] + '\n'})
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': END_MD})

    # Light formatting: clear exec counts only (keep sources as-is)
    for c in cells:
        if c.get('cell_type') == 'code':
            c['execution_count'] = None

    nb['cells'] = cells

    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False)

if __name__ == '__main__':
    enhance()
