# Petals to the Metal

School machine-learning project for the Kaggle competition **[Petals to the Metal — Flower Classification on TPU](https://www.kaggle.com/c/tpu-getting-started)** (`tpu-getting-started`).

## Goal

Classify flower images into the correct class. Work is graded in stages: claim the competition, ship a working dummy pipeline, then improve with transfer learning and dual Kaggle submissions.

Primary deliverables:

- Working Jupyter notebook(s) with sequential, reproducible code
- Kaggle submission CSVs (dummy, then two non-dummy models)
- Canvas uploads: notebook + screenshot of Kaggle scores

## Competition / data

| Item | Detail |
|------|--------|
| Kaggle competition slug | `tpu-getting-started` |
| Data format | TFRecord files |
| Local data path | `data/tpu-getting-started/` (gitignored — do not commit) |
| Expected pre-processing | Resize images, standardize pixel values; later add augmentation |

Data was previously downloaded with `kagglehub` into `./data`. That download script (`script.py`) was removed from the repo; recreate a small download helper if needed:

```python
from pathlib import Path
import kagglehub

data_dir = Path(__file__).resolve().parent / "data" / "tpu-getting-started"
data_dir.mkdir(parents=True, exist_ok=True)
path = kagglehub.competition_download("tpu-getting-started", output_dir=str(data_dir))
print("Path to competition files:", path)
```

Requires Kaggle credentials configured for `kagglehub` / the Kaggle API.

## Setup (teammates)

Each person needs a local Python env and the competition data. Do **not** commit `data/` or `.venv/`.

### 1. Python version

Use **Python 3.12** (or 3.11). System Python **3.14** does not have a TensorFlow wheel yet.

```bash
# macOS Homebrew example
brew install python@3.12
```

### 2. Create a virtual environment and install packages

From the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Current packages in `requirements.txt`:

- `tensorflow` — load TFRecords, training, inference
- `kagglehub` — download the competition dataset

### 3. Kaggle API credentials

1. Create a Kaggle account and accept the [competition rules](https://www.kaggle.com/c/tpu-getting-started/rules).
2. Account → Settings → API → **Create New Token** (downloads `kaggle.json`).
3. Place it where the tools expect it:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Download the dataset

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
import kagglehub

data_dir = Path("data") / "tpu-getting-started"
data_dir.mkdir(parents=True, exist_ok=True)
path = kagglehub.competition_download("tpu-getting-started", output_dir=str(data_dir))
print("Path to competition files:", path)
PY
```

You should end up with folders like `data/tpu-getting-started/tfrecords-jpeg-192x192/{train,val,test}/`.

### 5. Jupyter (optional but useful for Canvas)

```bash
source .venv/bin/activate
pip install jupyter
python -m ipykernel install --user --name=petal-to-metal --display-name="Python (petal-to-metal)"
```

Then pick that kernel in VS Code / Cursor / Jupyter when running the project notebook.

## Checklist & status

Track progress here so future agents know what is done vs. remaining.

### Competition claiming

- [x] Confirm "Petals to the Metal" is not already chosen by two teams on the class tracking form
- [x] Submit the form to lock in the team's choice

### Milestone 1 — Setup & baseline submission

**Data setup & pre-processing**

- [x] Download TFRecord dataset into the shared environment (`data/`)
- [ ] Initial pre-processing: image resizing, pixel standardization

**Dummy submission**

- [x] Find the most frequent flower class in the training data — **iris** (class id `67`, 782 / 12,753 train images)
- [ ] Create a submission file predicting that class for all test images
- [ ] Upload to Kaggle and confirm the pipeline works

**Canvas submission**

- [ ] Screenshot of Kaggle dummy score
- [ ] Clean notebook so all cells run sequentially
- [ ] Upload notebook + screenshot to Canvas

### Milestone 2 — Enhanced modeling & dual submissions

**Advanced pre-processing**

- [ ] Data augmentation (random flips, rotations, color tweaks)
- [ ] Training and validation pipelines for local model testing

**Model training & Kaggle submissions**

- [ ] Non-dummy #1: baseline transfer learning (e.g. ResNet50 or MobileNet) → test predictions → Kaggle submit
- [ ] Non-dummy #2: stronger / alternate model (e.g. EfficientNet or ConvNeXt) → test predictions → Kaggle submit

**Canvas submission**

- [ ] Both non-dummy scores visible on the Kaggle submission dashboard
- [ ] Notebook updated with all pre-processing and training code
- [ ] Updated notebook submitted to Canvas

## Repo state (for agents)

- Competition claiming is done; TFRecord data is downloaded under `data/tpu-getting-started/` (gitignored — do not commit).
- `.gitignore` ignores `data/` and `.venv/`.
- Local env: Python **3.12** venv + `requirements.txt` (see Setup). Default `python3` on some Macs is 3.14 and cannot install TensorFlow.
- There is currently no committed training notebook or model code — next useful work is Milestone 1 pre-processing (load TFRecords, resize/standardize), then majority-class dummy `submission.csv` predicting **iris / class 67**.
- Prefer a single clean Jupyter notebook that runs top-to-bottom for Canvas; keep heavy data off git.

## Suggested agent workflow

1. **Read this README** and update checklist boxes when something is finished.
2. **Confirm data** is present under `data/tpu-getting-started/` (or re-download).
3. **Milestone 1 first**: parse TFRecords → class distribution → constant-class dummy CSV → Kaggle upload → notebook cleanup.
4. **Milestone 2**: augmentation + train/val pipelines → two transfer-learning models → two Kaggle submissions → refresh notebook for Canvas.
5. Do not commit large datasets, model weights, or secrets (Kaggle API keys).

## Notes for teammates / graders

- Dummy submission exists only to prove the end-to-end path (data → CSV → Kaggle).
- Non-dummy work should show real training and two distinct model approaches, not two copies of the same run.
- Keep the notebook executable from a fresh kernel without hidden/out-of-order cells.
