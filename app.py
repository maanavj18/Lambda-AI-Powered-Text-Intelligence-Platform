"""Streamlit front end: type a review, see each model's sentiment prediction.

Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src.evaluate import clean, load_model, predict

# Models shown side by side on the Predict tab: display name -> predict function.
# A predict function takes the review text and returns ("positive" | "negative", confidence).
# A value of None reserves the column and shows a "not connected yet" note.
# TODO: when the second trained model is delivered, replace None with its predict function.
MODELS = {
    "Logistic Regression": predict,
    "Second model": None,
}

EXAMPLES = {
    "Positive example": "The coffee arrived fresh and tastes amazing. Rich flavor, no bitterness. Would buy again!",
    "Negative example": "Terrible. The chips were stale and tasted rancid. Very disappointed, I threw the whole bag away.",
    "Mixed example": "The flavor is great, but half the cookies arrived broken and the box was not sealed.",
}


def top_words(text: str, n: int = 5) -> list[tuple[str, float]]:
    """The words in this review that moved the Logistic Regression model most.

    Returns (word, score) pairs, strongest first. A positive score pushes toward
    'positive', a negative score toward 'negative'. The score is the word's count
    in the review times the model's weight for that word.
    """
    model = load_model()
    vectorizer, classifier = model.named_steps["vectorizer"], model.named_steps["classifier"]
    counts = vectorizer.transform([clean(text)])
    words = vectorizer.get_feature_names_out()
    scored = [(words[i], float(counts[0, i] * classifier.coef_[0][i])) for i in counts.indices]
    return sorted(scored, key=lambda pair: abs(pair[1]), reverse=True)[:n]


def show_prediction(label: str, confidence: float) -> None:
    # Never round up to a misleading "100%".
    shown = "over 99%" if confidence > 0.99 else f"{confidence:.0%}"
    if label == "positive":
        st.success(f"**Positive** 👍  \n{shown} confidence")
    else:
        st.error(f"**Negative** 👎  \n{shown} confidence")


st.set_page_config(page_title="Food Review Sentiment", page_icon="🍽️")

st.title("🍽️ Fine Food Review Sentiment")
st.write("Type or paste an Amazon fine food review and we'll predict whether it is positive or negative.")

predict_tab, about_tab = st.tabs(["Predict", "About"])

# ---------------------------------------------------------------- Predict tab
with predict_tab:
    st.caption("Try an example, or write your own review below.")
    for column, (label, text) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items()):
        if column.button(label, width="stretch"):
            st.session_state["review"] = text

    review = st.text_area(
        "Your review",
        key="review",
        height=160,
        placeholder="e.g. The coffee arrived fresh and tastes amazing. Would buy again!",
    )

    if st.button("Predict sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            for column, (name, predict_fn) in zip(st.columns(len(MODELS)), MODELS.items()):
                with column:
                    st.subheader(name)
                    if predict_fn is None:
                        st.info("Not connected yet. This model's prediction will appear here.")
                        continue
                    show_prediction(*predict_fn(review))

                    if name == "Logistic Regression":
                        words = top_words(review)
                        if words:
                            st.caption("Words that influenced this prediction most")
                            st.table(
                                pd.DataFrame(
                                    {"Pushes toward": ["Positive" if score > 0 else "Negative" for _, score in words]},
                                    index=pd.Index([word for word, _ in words], name="Word"),
                                )
                            )
                        else:
                            st.caption("None of these words appear in the model's vocabulary.")

# ------------------------------------------------------------------ About tab
with about_tab:
    st.subheader("What this app does")
    st.write(
        "It reads a food review and predicts whether the writer was happy (positive) or unhappy (negative) "
        "with the product. The model was trained on real Amazon reviews and learned which words tend to "
        "appear in each kind of review."
    )

    st.subheader("Data")
    st.write(
        "The model uses a 50,000-review sample of the "
        "[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset. "
        "Reviews with 4 or 5 stars are labeled positive, 1 or 2 stars negative, and 3-star reviews are left out."
    )

    st.subheader("Model")
    st.write(
        "The connected model is **Logistic Regression** on bag-of-words features. It counts which words appear "
        "in the review, gives every word a learned weight, and adds the weights up. Word order is ignored. "
        "A second model will be shown next to it for comparison once it is connected."
    )

    st.subheader("Limitations")
    st.markdown(
        "- Word order is ignored, so \"not good\" and sarcasm can be misread.\n"
        "- Words the model never saw in training have no effect on the prediction.\n"
        "- The confidence figure is the model's own estimate, not a guarantee."
    )
