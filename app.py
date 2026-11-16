"""Streamlit front end: type a review, get a sentiment prediction.

Run with:  streamlit run app.py
"""
import streamlit as st

from src.evaluate import predict

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
        result, confidence = predict(review)
        if result == "positive":
            st.success(f"Prediction: **Positive** 👍 ({confidence:.0%} confidence)")
        else:
            st.error(f"Prediction: **Negative** 👎 ({confidence:.0%} confidence)")

st.caption("Model: bag-of-words + logistic regression (models/sentiment.joblib).")
