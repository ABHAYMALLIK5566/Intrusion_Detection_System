import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / 'notebooks' / '02_inference_and_evaluation.ipynb'

REPLACE_PAIRS = [
    # Data paths
    ("pd.read_csv(\"KDDTest.txt\")", "pd.read_csv(\"../data/raw/KDDTest.txt\")"),
    ("pd.read_csv('KDDTest.txt')", "pd.read_csv('../data/raw/KDDTest.txt')"),
    ("pd.read_csv(\"KDDTrain.txt\")", "pd.read_csv(\"../data/raw/KDDTrain.txt\")"),
    ("pd.read_csv('KDDTrain.txt')", "pd.read_csv('../data/raw/KDDTrain.txt')"),
    # Artifacts paths
    ("joblib.load('model_lr.pkl')", "joblib.load('../artifacts/model_lr.pkl')"),
    ("joblib.load(\"model_lr.pkl\")", "joblib.load('../artifacts/model_lr.pkl')"),
    ("joblib.load('model_knn.pkl')", "joblib.load('../artifacts/model_knn.pkl')"),
    ("joblib.load('model_gnb.pkl')", "joblib.load('../artifacts/model_gnb.pkl')"),
    ("joblib.load('model_linear_svc.pkl')", "joblib.load('../artifacts/model_linear_svc.pkl')"),
    ("joblib.load('model_tdt.pkl')", "joblib.load('../artifacts/model_tdt.pkl')"),
    ("joblib.load('model_rf.pkl')", "joblib.load('../artifacts/model_rf.pkl')"),
    ("joblib.load('model_xg_r.pkl')", "joblib.load('../artifacts/model_xg_r.pkl')"),
    ("joblib.load('model_rrf.pkl')", "joblib.load('../artifacts/model_rrf.pkl')"),
    ("joblib.load('train_columns.pkl')", "joblib.load('../artifacts/train_columns.pkl')"),
    ("joblib.load('model_history.pkl')", "joblib.load('../artifacts/model_history.pkl')"),
]

DL_LOAD_CELL = (
    "### Deep Learning (loaded model)\n"
    "This section loads a pre-trained neural network from `../artifacts/` instead of training in-notebook."
)

DL_LOAD_CODE = (
    "# Load pre-trained Deep Learning model from artifacts\n"
    "import os\n"
    "import joblib\n"
    "import numpy as np\n"
    "import tensorflow as tf\n\n"
    "_dl_model = None\n"
    "_joblib_path = '../artifacts/model_dl.pkl'\n"
    "_savedmodel_dir = '../artifacts/model_dl'\n"
    "_h5_path = '../artifacts/model_dl.h5'\n\n"
    "try:\n"
    "    if os.path.exists(_joblib_path):\n"
    "        _dl_model = joblib.load(_joblib_path)\n"
    "        print('Loaded DL model from', _joblib_path)\n"
    "    elif os.path.isdir(_savedmodel_dir):\n"
    "        _dl_model = tf.keras.models.load_model(_savedmodel_dir)\n"
    "        print('Loaded DL model from', _savedmodel_dir)\n"
    "    elif os.path.exists(_h5_path):\n"
    "        _dl_model = tf.keras.models.load_model(_h5_path)\n"
    "        print('Loaded DL model from', _h5_path)\n"
    "except Exception as e:\n"
    "    print('Failed to load DL model:', e)\n\n"
    "if _dl_model is None:\n"
    "    print('Deep Learning model artifact not found. Place model_dl.pkl or model_dl(.h5) under ../artifacts/.')\n"
)

DL_EVAL_CODE = (
    "# Evaluate the loaded DL model if available\n"
    "from sklearn import metrics\n"
    "if _dl_model is not None:\n"
    "    x_eval = x_test.astype(np.float32)\n"
    "    y_eval = y_test.astype(np.int32)\n"
    "    y_pred_probs = _dl_model.predict(x_eval)\n"
    "    import numpy as _np\n"
    "    if _np.ndim(y_pred_probs) > 1:\n"
    "        y_pred_probs = _np.ravel(y_pred_probs)\n"
    "    y_pred = (y_pred_probs >= 0.5).astype(int)\n"
    "    print(f'DL Test Accuracy: {metrics.accuracy_score(y_eval, y_pred) * 100:.2f}%')\n"
    "    print(f'DL Test Precision: {metrics.precision_score(y_eval, y_pred) * 100:.2f}%')\n"
    "    print(f'DL Test Recall: {metrics.recall_score(y_eval, y_pred) * 100:.2f}%')\n"
    "    cm = metrics.confusion_matrix(y_eval, y_pred)\n"
    "    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['normal', 'attack'])\n"
    "    disp.plot()\n"
)

def to_source_lines(text: str):
    return [ln if ln.endswith('\n') else ln + '\n' for ln in text.splitlines()]


def process():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    inserted_loader = False
    inserted_eval = False

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            src_list = cell.get('source', [])
            src_text = ''.join(src_list) if isinstance(src_list, list) else src_list

            # Global replacements
            for old, new in REPLACE_PAIRS:
                if old in src_text:
                    src_text = src_text.replace(old, new)

            # Remove DL build/compile/fit cells
            if 'tf.keras.Sequential' in src_text or 'model = tf.keras.Sequential' in src_text:
                if not inserted_loader:
                    new_cells.append({
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': DL_LOAD_CELL,
                    })
                    new_cells.append({
                        'cell_type': 'code',
                        'metadata': {},
                        'execution_count': None,
                        'outputs': [],
                        'source': to_source_lines(DL_LOAD_CODE),
                    })
                    inserted_loader = True
                continue
            if 'model.compile(' in src_text:
                continue
            if 'model.fit(' in src_text:
                if not inserted_eval:
                    new_cells.append({
                        'cell_type': 'code',
                        'metadata': {},
                        'execution_count': None,
                        'outputs': [],
                        'source': to_source_lines(DL_EVAL_CODE),
                    })
                    inserted_eval = True
                continue

            # Keep other code cells with cleaned outputs and no exec count
            cell['source'] = to_source_lines(src_text)
            cell['execution_count'] = None
            cell['outputs'] = []
            new_cells.append(cell)
        else:
            new_cells.append(cell)

    # Ensure loader/eval present at the end if not inserted
    if not inserted_loader:
        new_cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': DL_LOAD_CELL})
        new_cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': to_source_lines(DL_LOAD_CODE)})
    if not inserted_eval:
        new_cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': to_source_lines(DL_EVAL_CODE)})

    nb['cells'] = new_cells

    # Clear notebook-level metadata execution state if any
    nb['metadata'].pop('widgets', None)

    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False)

if __name__ == '__main__':
    process()
