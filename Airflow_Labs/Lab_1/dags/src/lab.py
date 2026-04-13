import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import base64


def _serialize(obj):
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def _deserialize(b64_str):
    return pickle.loads(base64.b64decode(b64_str))


def load_data():
    """
    Loads the Iris dataset from CSV and returns it as a
    base64-encoded pickled DataFrame (JSON-safe for XCom).
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    print(f"Loaded Iris dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return _serialize(df)


def data_preprocessing(data_b64: str):
    """
    Splits features/target, does train/test split and standard scaling.
    Returns a base64-encoded dict with X_train, X_test, y_train, y_test.
    """
    df = _deserialize(data_b64)
    df = df.dropna()

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    payload = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "scaler": scaler,
    }
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    return _serialize(payload)


def build_save_model(data_b64: str, filename: str):
    """
    Trains a RandomForestClassifier on the training split and saves
    the model + scaler to disk. Returns the preprocessed data bundle
    (base64) so the next task can evaluate.
    """
    payload = _deserialize(data_b64)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(payload["X_train"], payload["y_train"])

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, filename)
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "scaler": payload["scaler"]}, f)

    print(f"Model saved to {model_path}")
    return data_b64


def evaluate_model(filename: str, data_b64: str):
    """
    Loads the saved model, evaluates on the test split, and also
    predicts on test.csv as a demo. Returns accuracy as a float.
    """
    model_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    clf = bundle["model"]
    scaler = bundle["scaler"]

    payload = _deserialize(data_b64)
    y_pred = clf.predict(payload["X_test"])
    acc = accuracy_score(payload["y_test"], y_pred)

    species_names = ["setosa", "versicolor", "virginica"]
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(payload["y_test"], y_pred, target_names=species_names))

    test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    test_scaled = scaler.transform(test_df)
    demo_pred = clf.predict(test_scaled)
    print(f"Demo prediction for test.csv: {species_names[demo_pred[0]]}")

    return float(acc)
