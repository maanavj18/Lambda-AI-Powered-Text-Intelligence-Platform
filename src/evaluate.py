"""Evaluate sentiment models and support the legacy app prediction entry point."""
import re
from functools import lru_cache
from pathlib import Path

from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATHS = [ROOT / "models" / "sentiment.joblib", ROOT / "notebooks" / "models" / "sentiment.joblib"]
LABELS = [0, 1]


def measure(actual, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=LABELS, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
    }


def save_loss_curve(history, selected_epoch, path):
    from matplotlib.figure import Figure

    epochs = [row["epoch"] for row in history]
    figure = Figure(figsize=(7.2, 3.8), layout="constrained")
    axis = figure.subplots()
    axis.plot(epochs, [row["training_loss"] for row in history], "o-", label="Training")
    axis.plot(epochs, [row["validation_loss"] for row in history], "s-", label="Validation")
    axis.axvline(selected_epoch, color="0.45", linestyle="--", label="Selected epoch")
    axis.set(xlabel="Epoch", ylabel="Binary cross-entropy", xticks=epochs)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=160)


@lru_cache(maxsize=1)
def load_model():
    """Return the fitted Pipeline (CountVectorizer -> LogisticRegression)."""
    path = next((p for p in MODEL_PATHS if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"sentiment.joblib not found in: {[str(p) for p in MODEL_PATHS]}")
    return load(path)


def clean(text: str) -> str:
    """Same cleaning as training: strip HTML tags."""
    return re.sub(r"<.*?>", " ", text)


def predict(text: str) -> tuple[str, float]:
    """Return ('positive'|'negative', confidence) for one review."""
    proba = load_model().predict_proba([clean(text)])[0]
    label = int(proba.argmax())
    return ("positive" if label == 1 else "negative"), float(proba[label])
