"""Basic Streamlit front end: type a review, get a sentiment prediction.

Run with:  streamlit run app.py

This file is self-contained (no other files needed).
The model is NOT connected yet. predict() below uses the majority-class
baseline as a placeholder; swap its body for the real model call later.
"""
from collections import Counter

import streamlit as st


def majority_class_baseline(train_labels, test_inputs):
    """Predict the most common training label for every input (placeholder model)."""
    counts = Counter(train_labels)
    majority = counts.most_common(1)[0][0]
    return [majority] * len(test_inputs)


# Placeholder "training labels" (Milestone 2 split is ~84.7% positive).
# Replace with real model loading once the ML Engineer's artifacts are ready.
PLACEHOLDER_TRAIN_LABELS = [1] * 847 + [0] * 153


def predict(review_text: str) -> str:
    """Return 'positive' or 'negative' for one review.

    TODO: replace with the real pipeline (clean text -> vectorize -> model).
    """
    label = majority_class_baseline(PLACEHOLDER_TRAIN_LABELS, [review_text])[0]
    return "positive" if label == 1 else "negative"


st.set_page_config(page_title="Food Review Sentiment", page_icon="🍽️")

st.title("🍽️ Fine Food Review Sentiment")
st.write("Type or paste an Amazon fine food review and we'll predict whether it is positive or negative.")

review = st.text_area(
    "Your review",
    height=180,
    placeholder="e.g. The coffee arrived fresh and tastes amazing. Would buy again!",
)

if st.button("Predict sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        result = predict(review)
        if result == "positive":
            st.success("Prediction: **Positive** 👍")
        else:
            st.error("Prediction: **Negative** 👎")

st.caption("Demo build: predictions currently come from a placeholder baseline model.")