from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from data import load_data, split_data

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "wine_model.pkl"

def fit_model(X_train, y_train):
    """
    Train a Logistic Regression classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    lr_classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=12),
    )
    lr_classifier.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_classifier, MODEL_PATH)

if __name__ == "__main__":
    X, y = load_data()
    X_train, _, y_train, _ = split_data(X, y)
    fit_model(X_train, y_train)
