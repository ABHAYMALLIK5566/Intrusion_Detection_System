import os
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / 'notebooks' / '02_inference_and_evaluation.ipynb'

ART_DIR = Path('..') / 'artifacts'

REPLACE_LOADS = {
    "joblib.load('model_lr.pkl')": "joblib.load('../artifacts/model_lr.pkl')",
    "joblib.load(\"model_lr.pkl\")": "joblib.load('../artifacts/model_lr.pkl')",
    "joblib.load('model_knn.pkl')": "joblib.load('../artifacts/model_knn.pkl')",
    "joblib.load('model_gnb.pkl')": "joblib.load('../artifacts/model_gnb.pkl')",
    "joblib.load('model_linear_svc.pkl')": "joblib.load('../artifacts/model_linear_svc.pkl')",
    "joblib.load('model_tdt.pkl')": "joblib.load('../artifacts/model_tdt.pkl')",
    "joblib.load('model_rf.pkl')": "joblib.load('../artifacts/model_rf.pkl')",
    "joblib.load('model_xg_r.pkl')": "joblib.load('../artifacts/model_xg_r.pkl')",
    "joblib.load('model_rrf.pkl')": "joblib.load('../artifacts/model_rrf.pkl')",
    "joblib.load('train_columns.pkl')": "joblib.load('../artifacts/train_columns.pkl')",
    "joblib.load('model_history.pkl')": "joblib.load('../artifacts/model_history.pkl')",
}

DL_LOAD_CELL = """
# Load pre-trained Deep Learning model from artifacts
import os
import joblib
import numpy as np
import tensorflow as tf

# Try multiple common artifact formats
_dl_model = None
_joblib_path = '../artifacts/model_dl.pkl'
_savedmodel_dir = '../artifacts/model_dl'
_h5_path = '../artifacts/model_dl.h5'

try:
    if os.path.exists(_joblib_path):
        _dl_model = joblib.load(_joblib_path)
        print('Loaded DL model from', _joblib_path)
    elif os.path.isdir(_savedmodel_dir):
        _dl_model = tf.keras.models.load_model(_savedmodel_dir)
        print('Loaded DL model from', _savedmodel_dir)
    elif os.path.exists(_h5_path):
        _dl_model = tf.keras.models.load_model(_h5_path)
        print('Loaded DL model from', _h5_path)
except Exception as e:
    print('Failed to load DL model:', e)

if _dl_model is None:
    print('Deep Learning model artifact not found in ../artifacts/. Provide model_dl.pkl or model_dl(.h5).')
""".strip()

DL_EVAL_CELL = """
# Evaluate the loaded DL model if available
from sklearn import metrics
if _dl_model is not None:
    # Ensure correct dtypes
    x_eval = x_test.astype(np.float32)
    y_eval = y_test.astype(np.int32)
    # Predict probabilities or logits depending on model
    try:
        y_pred_probs = _dl_model.predict(x_eval)
    except Exception:
        # Some wrappers expose decision_function; fallback to predict
        y_pred_probs = _dl_model.predict(x_eval)
    # Handle shapes
    if y_pred_probs.ndim > 1:
        y_pred_probs = y_pred_probs.ravel()
    # Threshold at 0.5 for binary classification
    y_pred = (y_pred_probs >= 0.5).astype(int)

    print(f'DL Test Accuracy: {metrics.accuracy_score(y_eval, y_pred) * 100:.2f}%')
    print(f'DL Test Precision: {metrics.precision_score(y_eval, y_pred) * 100:.2f}%')
    print(f'DL Test Recall: {metrics.recall_score(y_eval, y_pred) * 100:.2f}%')

    cm = metrics.confusion_matrix(y_eval, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['normal', 'attack'])
    disp.plot()
""".strip()

def main():
    nb = nbf.read(NB_PATH, as_version=nbf.NO_CONVERT)

    new_cells = []
    dl_load_inserted = False
    dl_eval_inserted = False

    for cell in nb.cells:
        if cell.cell_type == 'code':
            src = cell.source
            # Path fixes for artifacts
            for old, new in REPLACE_LOADS.items():
                if old in src:
                    src = src.replace(old, new)
            # Remove DL build/compile/fit cells
            if 'tf.keras.Sequential' in src or 'model = tf.keras.Sequential' in src:
                if not dl_load_inserted:
                    new_cells.append(nbf.v4.new_code_cell(DL_LOAD_CELL))
                    dl_load_inserted = True
                continue  # skip original build cell
            if 'model.compile(' in src:
                # drop compile cell
                continue
            if 'model.fit(' in src:
                # replace fit with evaluation cell
                if not dl_eval_inserted:
                    new_cells.append(nbf.v4.new_code_cell(DL_EVAL_CELL))
                    dl_eval_inserted = True
                continue
            # Update source in other code cells
            cell.source = src
            new_cells.append(cell)
        else:
            new_cells.append(cell)

    # If we removed DL cells but did not add loaders/eval (e.g., not found), ensure loader present for clarity
    if not dl_load_inserted:
        new_cells.append(nbf.v4.new_markdown_cell('### Deep Learning model\nThis notebook expects a pre-trained DL model in `../artifacts/` named `model_dl.pkl` (or Keras `model_dl`/`model_dl.h5`).'))
        new_cells.append(nbf.v4.new_code_cell(DL_LOAD_CELL))
    if not dl_eval_inserted:
        new_cells.append(nbf.v4.new_code_cell(DL_EVAL_CELL))

    nb.cells = new_cells
    nbf.write(nb, NB_PATH)

if __name__ == '__main__':
    main()
