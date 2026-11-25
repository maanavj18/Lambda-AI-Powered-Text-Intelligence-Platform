"""Connect the Streamlit app to the team's trained models in models/.

Labels come from the ML engineer's handoff API, `src.predict.load_predictor`.
That API returns only 0 or 1, so this module also reads each model's
probability of "positive" from the same saved files to show a confidence.
"""
import json
from functools import lru_cache
from pathlib import Path

from joblib import load

from src.predict import load_predictor
from src.preprocessing import clean_review

MODELS_DIR = Path(__file__).resolve().parent / "models"
METRICS_PATH = MODELS_DIR / "comparison_metrics.json"
LOSS_CURVE_PATH = MODELS_DIR / "lstm_loss.png"

# Display name (as used in comparison_metrics.json) -> name accepted by load_predictor
MODELS = {
    "Logistic Regression": "logistic_regression",
    "Multinomial Naive Bayes": "naive_bayes",
    "LSTM": "lstm",
}
BAG_OF_WORDS_MODELS = ("logistic_regression", "naive_bayes")


@lru_cache(maxsize=None)
def _pipeline(key: str):
    """The fitted CountVectorizer -> classifier pipeline for a bag-of-words model."""
    return load(MODELS_DIR / f"{key}.joblib")


@lru_cache(maxsize=1)
def _lstm():
    """The LSTM and its vocabulary, loaded the same way as src.predict."""
    import torch

    from src.lstm import SentimentLSTM

    checkpoint = torch.load(MODELS_DIR / "lstm.pt", map_location="cpu", weights_only=True)
    vocabulary = checkpoint["vocabulary"]
    model = SentimentLSTM(len(vocabulary) + 2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocabulary


def positive_probability(name: str, text: str) -> float:
    """The model's estimated probability that the review is positive."""
    key = MODELS[name]
    cleaned = clean_review(text)
    if key in BAG_OF_WORDS_MODELS:
        return float(_pipeline(key).predict_proba([cleaned])[0][1])

    import torch

    from src.lstm import encode_reviews

    model, vocabulary = _lstm()
    with torch.inference_mode():
        return float(torch.sigmoid(model(encode_reviews([cleaned], vocabulary))).item())


def predict(name: str, text: str) -> tuple[str, float]:
    """Return ("positive" | "negative", confidence) for one review."""
    label = load_predictor(MODELS[name])(text)
    probability = positive_probability(name, text)
    return ("positive", probability) if label == 1 else ("negative", 1 - probability)


def top_words(name: str, text: str, n: int = 5) -> list[tuple[str, float]] | None:
    """The words in this review that moved a bag-of-words model most.

    Returns (word, score) pairs, strongest first: a positive score pushes toward
    "positive", a negative score toward "negative". Returns None for the LSTM,
    which reads words in order and has no per-word weights.
    """
    key = MODELS[name]
    if key not in BAG_OF_WORDS_MODELS:
        return None
    pipeline = _pipeline(key)
    vectorizer, classifier = pipeline.named_steps["vectorizer"], pipeline.named_steps["classifier"]
    if hasattr(classifier, "coef_"):  # Logistic Regression: learned weight per word
        weights = classifier.coef_[0]
    else:  # Naive Bayes: how much likelier the word is in positive than negative reviews
        weights = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    counts = vectorizer.transform([clean_review(text)])
    words = vectorizer.get_feature_names_out()
    scored = [(words[i], float(counts[0, i] * weights[i])) for i in counts.indices]
    return sorted(scored, key=lambda pair: abs(pair[1]), reverse=True)[:n]


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    """Held-out test results written by src/train.py."""
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
