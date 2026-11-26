"""Streamlit front end: classify a review, then read about the data and model results.

Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

import app_charts as charts
import app_data as data
import app_models as models

EXAMPLES = {
    "Positive example": "The coffee arrived fresh and tastes amazing. Rich flavor, no bitterness. Would buy again!",
    "Negative example": "Terrible. The chips were stale and tasted rancid. Very disappointed, I threw the whole bag away.",
    "Mixed example": "The flavor is great, but half the cookies arrived broken and the box was not sealed.",
    "Negation example": "Not good. The flavor is not great and it is not fresh.",
}


def show_prediction(label: str, confidence: float) -> None:
    # Never round up to a misleading "100%".
    shown = "over 99%" if confidence > 0.99 else f"{confidence:.0%}"
    if label == "positive":
        st.success(f"**Positive** 👍  \n{shown} confidence")
    else:
        st.error(f"**Negative** 👎  \n{shown} confidence")


def show_top_words(name: str, review: str) -> None:
    words = models.top_words(name, review)
    if words is None:
        st.caption("The LSTM reads the words in order, so it has no single-word breakdown.")
    elif not words:
        st.caption("None of these words or phrases appear in the model's vocabulary.")
    else:
        st.caption("Words or phrases that influenced this prediction most")
        st.table(
            pd.DataFrame(
                {"Pushes toward": ["Positive" if score > 0 else "Negative" for _, score in words]},
                index=pd.Index([word for word, _ in words], name="Phrase"),
            )
        )


st.set_page_config(page_title="Food Review Sentiment", page_icon="🍽️", layout="wide")

st.title("🍽️ Fine Food Review Sentiment")
st.write(
    "Type or paste an Amazon fine food review and compare whether each model "
    "thinks it is positive or negative."
)

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
        height=140,
        placeholder="e.g. The coffee arrived fresh and tastes amazing. Would buy again!",
    )

    if st.button("Predict sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Running the models..."):
                results = {name: models.predict(name, review) for name in models.MODELS}
            for column, (name, (label, confidence)) in zip(st.columns(len(results)), results.items()):
                with column:
                    st.subheader(name)
                    show_prediction(label, confidence)
                    show_top_words(name, review)

            labels = {label for label, _ in results.values()}
            if len(labels) == 1:
                st.info(f"All {len(results)} models agree: this review is **{labels.pop()}**.")
            else:
                st.info(
                    "The models disagree on this review. Mixed or sarcastic reviews are where they differ most."
                )

# ------------------------------------------------------------------ About tab
with about_tab:
    metrics = models.load_metrics()
    results = metrics["models"]
    splits = metrics["splits"]

    st.subheader("What this app does")
    st.write(
        "It reads a food review and predicts whether the writer was happy (positive) or unhappy (negative) "
        "with the product. Four models, each trained on real Amazon reviews, make the prediction "
        "independently so their answers can be compared."
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
    c2.metric("Used after cleaning", f"{data.CLEANED_ROWS:,}")
    c3.metric("Positive", f"{positive_share:.0%}")
    c4.metric("Avg. length", f"{data.AVG_WORDS:.0f} words", help=f"Median: {data.MEDIAN_WORDS} words")

    chart_column, quality_column = st.columns([3, 2], gap="large")
    with chart_column:
        st.altair_chart(charts.star_distribution(data.SCORE_COUNTS), width="stretch")
        st.caption(
            f"Most reviews are 5 stars, so the classes are imbalanced: about {positive_share:.0%} of labeled "
            "reviews are positive. A model that always says \"positive\" would already be right that often."
        )
    with quality_column:
        st.markdown("**Data quality issues**")
        st.table(pd.DataFrame(
            {
                "Issue": [
                    "Duplicate reviews (same user, time and text)",
                    *[f"Missing {column}" for column in data.MISSING_VALUES],
                ],
                "Count": [f"{data.DUPLICATE_REVIEWS:,}", *[f"{n:,}" for n in data.MISSING_VALUES.values()]],
                "How it was handled": ["Removed", "Replaced with empty text", "Not used by the models"],
            }
        ).set_index("Issue"))

    st.subheader("Models")
    st.markdown(
        "- **Logistic Regression** counts which words appear in the review (bag of words), gives every word "
        "a learned weight, and adds the weights up.\n"
        "- **Multinomial Naive Bayes** uses the same word counts and compares how often each word shows up "
        "in positive versus negative reviews.\n"
        "- **Improved Naive Bayes** does the same comparison, but on phrases of one to three words, and it "
        "only checks whether each phrase appears rather than how many times.\n"
        "- **LSTM** is a neural network that reads the review word by word, in order, so it can use context "
        "such as \"not\" before \"good\".\n\n"
        "Every model sees the review summary and text together, with HTML tags removed."
    )

    st.subheader("How the models performed")
    def count(split: str) -> int:
        return splits[split]["negative"] + splits[split]["positive"]

    st.write(
        f"Logistic Regression, Multinomial Naive Bayes, and the LSTM learned from {count('training'):,} reviews "
        f"and were tuned on {count('validation'):,} validation reviews. Improved Naive Bayes follows the notebook: "
        f"its settings were chosen with 5-fold cross-validation on those {count('training') + count('validation'):,} "
        f"reviews together. Every model was then scored on the same {count('test'):,} held-out test reviews. "
        "The majority baseline always answers \"positive\"; a model has to beat it to show it is learning anything."
    )
    table = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": r["accuracy"],
            "Precision": r["macro_precision"],
            "Recall": r["macro_recall"],
            "F1": r["macro_f1"],
            "Training time (s)": r["training_seconds"],
        }
        for name, r in results.items()
    ]).set_index("Model")
    score_format = {column: "{:.3f}" for column in ["Accuracy", "Precision", "Recall", "F1"]}
    st.table(table.style.format({**score_format, "Training time (s)": "{:.2f}"}))
    best = max((name for name in results if "baseline" not in name.lower()), key=lambda n: results[n]["macro_f1"])
    st.caption(
        f"Precision, recall and F1 are macro averages, so negative and positive reviews count equally. "
        f"**{best}** has the highest F1. Training time is in seconds on one laptop CPU."
    )

    st.markdown("**Confusion matrices**")
    st.caption(
        "Rows are the true label, columns the prediction. The diagonal is correct; the other two cells are "
        "mistakes. Hover over a cell for details."
    )
    for column, (name, r) in zip(st.columns(len(results)), results.items()):
        with column:
            st.altair_chart(charts.confusion_matrix(r["confusion_matrix"], name), width="stretch")

    if models.LOSS_CURVE_PATH.exists():
        lstm = results.get("LSTM", {})
        curve_column, text_column = st.columns([3, 2], gap="large")
        with curve_column:
            st.markdown("**LSTM training and validation loss**")
            st.image(str(models.LOSS_CURVE_PATH), width="stretch")
        with text_column:
            st.markdown("&nbsp;")
            st.caption(
                "Loss measures how wrong the model's predictions are, so lower is better. Training loss keeps "
                "falling, but validation loss stops improving after a few epochs: past that point the model is "
                "memorizing the training reviews rather than learning patterns that carry over. "
                + (
                    f"The saved model is from epoch {lstm['selected_epoch']}, the epoch with the best "
                    "validation F1."
                    if "selected_epoch" in lstm
                    else ""
                )
            )

    st.subheader("Limitations")
    st.markdown(
        "- Logistic Regression and Multinomial Naive Bayes ignore word order, so \"not good\" and sarcasm "
        "can be misread. Improved Naive Bayes keeps phrases of up to three words, but longer context can "
        "still be missed.\n"
        "- The LSTM reads only the first 128 words of a review, so evidence near the end of a long review "
        "can be missed.\n"
        "- Words a model never saw in training have no effect on its prediction.\n"
        "- The confidence figure is the model's own estimate, not a guarantee."
    )
