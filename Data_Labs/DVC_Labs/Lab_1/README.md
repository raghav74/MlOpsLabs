# DVC Lab 1 -- MNIST Pipeline with Local Remote

## Introduction

[DVC (Data Version Control)](https://dvc.org/) is an open-source tool for versioning data and building reproducible ML pipelines. In this lab you will:

- Build a **three-stage DVC pipeline** (data preparation, training, evaluation) for MNIST digit classification.
- Use a **local directory** as the DVC remote -- no cloud credentials required.
- Experiment with hyperparameters and use `dvc metrics diff` to compare results across runs.

### Pipeline Overview

```
prepare_data  ──►  train  ──►  evaluate
```

| Stage | Script | Inputs | Outputs |
|-------|--------|--------|---------|
| `prepare_data` | `src/prepare_data.py` | `params.yaml` | `data/processed/` |
| `train` | `src/train.py` | `data/processed/`, `params.yaml` | `models/model.pt` |
| `evaluate` | `src/evaluate.py` | `data/processed/`, `models/model.pt` | `metrics/metrics.json` |

---

## Prerequisites

1. **Anaconda** installed on your machine.
2. Activate the `mlops_project` environment:

```bash
conda activate mlops_project
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Step 1 -- Initialize DVC

Navigate to the lab folder and initialize DVC. Because this lab lives inside a subdirectory of the git repo (not at the repo root), use the `--subdir` flag:

```bash
cd Data_Labs/DVC_Labs/Lab_1
dvc init --subdir
```

This creates a `.dvc/` directory that stores DVC configuration and cache.

> **Note:** The `--subdir` flag is required when running `dvc init` in a subdirectory of a git repository. If DVC was already initialized (`.dvc/` exists in this folder), you can skip this step.

---

## Step 2 -- Set Up a Local Remote

Create a directory that will act as the DVC remote storage, then register it with DVC:

```bash
mkdir dvc_remote
dvc remote add -d myremote dvc_remote
```

Verify the configuration:

```bash
dvc remote list
# myremote  dvc_remote
```

The `-d` flag sets this as the **default** remote. All `dvc push` and `dvc pull` commands will target it automatically.

---

## Step 3 -- Understand the Pipeline

### `params.yaml`

Central configuration file for the pipeline:

```yaml
data:
  seed: 42

train:
  epochs: 5
  batch_size: 64
  learning_rate: 0.001
```

### `dvc.yaml`

Defines the three pipeline stages, their dependencies, outputs, and metrics:

```yaml
stages:
  prepare_data:
    cmd: python src/prepare_data.py
    deps:
      - src/prepare_data.py
    params:
      - data
    outs:
      - data/processed

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed
    params:
      - train
    outs:
      - models/model.pt

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pt
      - data/processed
    metrics:
      - metrics/metrics.json:
          cache: false
```

DVC automatically builds a DAG from these dependencies so stages run in the correct order.

---

## Step 4 -- Run the Pipeline

Execute the entire pipeline with a single command:

```bash
dvc repro
```

DVC will run each stage in order:

1. **prepare_data** -- Downloads MNIST via torchvision and saves processed tensors to `data/processed/`.
2. **train** -- Trains a small CNN on the training set and saves the model to `models/model.pt`.
3. **evaluate** -- Evaluates the model on the test set and writes accuracy/loss to `metrics/metrics.json`.

View the metrics:

```bash
dvc metrics show
```

---

## Step 5 -- Push Data to the Local Remote

Push all DVC-tracked outputs to the local remote:

```bash
dvc push
```

This copies the cached data (`data/processed/`, `models/model.pt`) into the `dvc_remote/` directory. In a real project this would be S3, GCS, or Azure Blob -- the workflow is identical.

---

## Step 6 -- Experiment with Versioning

1. Open `params.yaml` and change a hyperparameter, for example:

```yaml
train:
  epochs: 10          # was 5
  batch_size: 64
  learning_rate: 0.001
```

2. Re-run the pipeline (DVC only re-runs stages whose dependencies changed):

```bash
dvc repro
```

3. Compare metrics between the old and new runs:

```bash
dvc metrics diff
```

You will see a table showing the change in accuracy and loss.

4. Commit the updated `dvc.lock` and `params.yaml` to Git to record this experiment:

```bash
git add dvc.lock params.yaml metrics/metrics.json
git commit -m "experiment: train for 10 epochs"
```

---

## Step 7 -- Pull Data from the Remote

Simulate a fresh clone by deleting the local cache and outputs:

```bash
rm -rf .dvc/cache data/processed models
```

Then pull everything back from the remote:

```bash
dvc pull
```

All data and model files are restored from `dvc_remote/`.

---

## Project Structure

```
Lab_1/
├── README.md
├── requirements.txt
├── params.yaml
├── dvc.yaml
├── dvc.lock              # auto-generated after dvc repro
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   └── processed/        # DVC-tracked
├── models/
│   └── model.pt          # DVC-tracked
├── metrics/
│   └── metrics.json      # DVC metric (not cached)
├── dvc_remote/           # local DVC remote storage
└── .dvc/
    ├── config
    └── cache/
```

---

## Useful DVC Commands Reference

| Command | Description |
|---------|-------------|
| `dvc init` | Initialize DVC in a project |
| `dvc remote add -d <name> <path>` | Add a default remote |
| `dvc repro` | Reproduce the pipeline |
| `dvc push` | Push tracked data to remote |
| `dvc pull` | Pull tracked data from remote |
| `dvc metrics show` | Display current metrics |
| `dvc metrics diff` | Compare metrics between commits |
| `dvc dag` | Visualize the pipeline DAG |
| `dvc status` | Show which stages are outdated |
