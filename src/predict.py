"""Load the fitted comparison models for prediction on raw reviews."""

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from joblib import load

from .preprocessing import clean_review


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
CLASSICAL_MODELS = ("logistic_regression", "naive_bayes")


@lru_cache(maxsize=3)
def load_predictor(name: str) -> Callable[[str, str], int]:
    """Return a predictor accepting review text and an optional summary."""
    if name in CLASSICAL_MODELS:
        pipeline = load(MODELS_DIR / f"{name}.joblib")

        def predict(text: str, summary: str = "") -> int:
            return int(pipeline.predict([clean_review(text, summary)])[0])

        return predict

    if name == "lstm":
        import torch

        from .lstm import SentimentLSTM, encode_reviews

        checkpoint = torch.load(MODELS_DIR / "lstm.pt", map_location="cpu", weights_only=True)
        vocabulary = checkpoint["vocabulary"]
        model = SentimentLSTM(len(vocabulary) + 2)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        def predict_lstm(text: str, summary: str = "") -> int:
            tokens = encode_reviews([clean_review(text, summary)], vocabulary)
            with torch.inference_mode():
                return int(model(tokens).item() >= 0)

        return predict_lstm

    raise ValueError(f"Unknown model {name!r}. Choose logistic_regression, naive_bayes, or lstm.")
