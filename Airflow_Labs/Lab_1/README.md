# Airflow Lab 1 — Iris Classification Pipeline

An Airflow DAG that trains and evaluates a Random Forest classifier on the Iris dataset.

## Model and Data

- **Dataset:** Iris — 150 samples, 4 numeric features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`), 3 classes (0 = setosa, 1 = versicolor, 2 = virginica)
- **Model:** `RandomForestClassifier` (100 estimators)
- **Pipeline:** `load_data` → `data_preprocessing` (StandardScaler, 80/20 train/test split) → `build_save_model` → `evaluate_model` (accuracy + classification report)
- **Dependencies:** `pandas`, `scikit-learn`

## Project Structure

```
Lab_1/
├── dags/
│   ├── airflow.py              # DAG definition (Iris_Classification_Lab1)
│   ├── src/
│   │   └── lab.py              # Pipeline functions
│   └── data/
│       ├── file.csv            # Iris dataset
│       └── test.csv            # Single sample for demo prediction
├── docker-compose.yaml         # Airflow + Postgres (LocalExecutor)
└── README.md
```

## Setup and Run

**Prerequisites:** Docker (4GB+ memory allocated).

```bash
echo "AIRFLOW_UID=$(id -u)" > .env
docker compose up airflow-init
docker compose up
```

Open `localhost:8080` (admin / admin), trigger the `Iris_Classification_Lab1` DAG, and check the `evaluate_model_task` logs for accuracy and classification report output.
