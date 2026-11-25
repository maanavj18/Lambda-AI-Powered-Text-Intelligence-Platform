# Lamda: AI-Powered Text Intelligence Platform

Sentiment analysis of Amazon Fine Food Reviews, from data to a deployed app.

CSE 3521: Survey of Artificial Intelligence I | Autumn 2026 | Flagship Group Project

## Team Members

- Landon
- Maanav Jajodia
- Irene
- Supra
- Yonal

## Project Description

Lamda is a text intelligence system that predicts the sentiment of Amazon food reviews. The team works from a shared sample of 50,000 reviews drawn from the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset (568,454 reviews with text, 1–5 star ratings, timestamps, and helpfulness votes). Review text and summaries are the inputs. Star ratings are mapped to a sentiment label the team chooses, such as positive/negative or a three-class scheme.

The comparison uses bag-of-words counts for Multinomial Naive Bayes and Logistic Regression, and trainable word embeddings for the LSTM. All three models are evaluated against a majority-class baseline on the same held-out reviews.

## How to Run

Use Python 3.13, the version used for the comparison. Run all commands from the repository root.

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Get the trained model.** The current app loads `models/sentiment.joblib` or `notebooks/models/sentiment.joblib`. Run the notebook through the first model export and copy its `bow-lr.joblib` to `models/sentiment.joblib`. [Data and model notebook](notebooks/1_data_clean.ipynb)

4. **Launch the app**

   ```bash
   streamlit run app.py
   ```

   Streamlit opens the app in your browser (default http://localhost:8501). Type a review and click **Predict sentiment**.

## Reproduce the model comparison

1. Set up the virtual environment and install the pinned dependencies above. The recorded run used the CPU build of PyTorch.
2. Put `Reviews.csv` in `data/`. If the file is absent, the training script uses KaggleHub to download dataset version 2. To provide a CSV at a different path, pass `--reviews path/to/Reviews.csv`. [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
3. Train and evaluate all three models with `python -m src.train`. The default run uses four CPU threads, at most 12 LSTM epochs, and early stopping after three epochs without improvement in validation macro F1.

The command writes fitted Naive Bayes and Logistic Regression pipelines, an LSTM checkpoint, the loss curve, and full-precision metrics to `models/`. Use `--models-dir` to write the outputs to another directory. The comparison summarizes held-out scores, confusion matrices, and model settings. [Model comparison](ml_engineer_model_comparison.md) · [Run metrics](models/comparison_metrics.json)

For inference, call `load_predictor` from `src.predict` with `"logistic_regression"`, `"naive_bayes"`, or `"lstm"`. The returned function accepts a review and an optional summary and returns `0` for negative or `1` for positive.
