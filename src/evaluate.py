"""Load the saved sentiment pipeline and predict on raw review text."""
import re
from functools import lru_cache
from pathlib import Path

from joblib import load

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATHS = [ROOT / "models" / "sentiment.joblib", ROOT / "notebooks" / "models" / "sentiment.joblib"]

@lru_cache(maxsize=1)
def load_model(model: str | None = None):
    """Return the fitted Pipeline (CountVectorizer -> LogisticRegression)."""
    path = ROOT / "models" / f"{model}.joblib" if model else None
    if path is None:
        path = next((p for p in MODEL_PATHS if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"sentiment.joblib not found in: {[str(p) for p in MODEL_PATHS]}")
    return load(path)


def clean(text: str) -> str:
    """Same cleaning as training: strip HTML tags."""
    return re.sub(r"<.*?>", " ", text)


def predict(text: str, model: str | None = None) -> tuple[str, float]:
    """Return ('positive'|'negative', confidence) for one review."""
    proba = load_model(model).predict_proba([clean(text)])[0]
    label = int(proba.argmax())
    return ("positive" if label == 1 else "negative"), float(proba[label])
