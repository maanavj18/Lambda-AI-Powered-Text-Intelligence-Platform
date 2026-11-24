"""Streamlit front end: classify a review, then read about the data and model results.

Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

import app_charts as charts
import app_data as data
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
    labeled = sum(data.LABEL_COUNTS.values())
    positive_share = data.LABEL_COUNTS["Positive"] / labeled
    st.write(
        f"We use a {data.SAMPLE_ROWS:,}-review random sample of the "
        "[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset "
        f"({data.FULL_DATASET_ROWS:,} reviews in total). Reviews with 4 or 5 stars are labeled positive, "
        "1 or 2 stars negative, and 3-star reviews are left out."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews in sample", f"{data.SAMPLE_ROWS:,}")
    c2.metric("Used after cleaning", f"{data.CLEANING_STEPS['Duplicates removed']:,}")
    c3.metric("Positive", f"{positive_share:.0%}")
    c4.metric("Avg. length", f"{data.AVG_WORDS:.0f} words", help=f"Median: {data.MEDIAN_WORDS} words")

    st.altair_chart(charts.star_distribution(data.SCORE_COUNTS), width="stretch")
    st.caption(
        f"Most reviews are 5 stars, so the classes are imbalanced: about {positive_share:.0%} of labeled "
        "reviews are positive. A model that always says \"positive\" would already be right that often."
    )

    st.markdown("**Data quality issues**")
    st.table(pd.DataFrame(
        {
            "Issue": [
                "Duplicate reviews (same user, time and text)",
                *[f"Missing {column}" for column in data.MISSING_VALUES],
            ],
            "Count": [f"{data.DUPLICATE_REVIEWS:,}", *[f"{n:,}" for n in data.MISSING_VALUES.values()]],
            "How it was handled": ["Removed", "Replaced with empty text", "Not used by the model"],
        }
    ).set_index("Issue"))

    st.subheader("Model")
    st.write(
        "The connected model is **Logistic Regression** on bag-of-words features. It counts which words appear "
        "in the review, gives every word a learned weight, and adds the weights up. Word order is ignored. "
        "A second model will be shown next to it for comparison once it is connected."
    )

    st.subheader("How the models performed")
    train_total = sum(data.SPLIT["Training set"].values())
    test_total = sum(data.SPLIT["Test set"].values())
    st.write(
        f"Models were trained on {train_total:,} reviews (80%) and scored on {test_total:,} held-out reviews "
        "(20%) they never saw during training. The majority baseline always answers \"positive\"; a model has "
        "to beat it to show it is learning anything."
    )
    results = data.MODEL_RESULTS
    table = pd.DataFrame([
        {"Model": name, "Accuracy": r["accuracy"], "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"]}
        for name, r in results.items()
    ]).set_index("Model")
    st.table(table.style.format("{:.3f}"))
    st.caption(
        "Precision, recall and F1 are macro averages, so negative and positive reviews count equally. "
        "That is why the baseline's accuracy looks high while its F1 is low."
    )

    st.markdown("**Confusion matrices**")
    st.caption("Rows are the true label, columns the prediction. The diagonal is correct; the other two cells are mistakes.")
    for column, (name, r) in zip(st.columns(len(results)), results.items()):
        with column:
            st.altair_chart(charts.confusion_matrix(r["confusion_matrix"], name), width="stretch")
    st.caption("Logistic Regression's test results will be added here once they are reported.")

    st.subheader("Limitations")
    st.markdown(
        "- Word order is ignored, so \"not good\" and sarcasm can be misread.\n"
        "- Words the model never saw in training have no effect on the prediction.\n"
        "- The confidence figure is the model's own estimate, not a guarantee."
    )
