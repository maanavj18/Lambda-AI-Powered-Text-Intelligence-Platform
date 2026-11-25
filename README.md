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

## Data and Cleaning

All the data is prepared and cleaned in [notebooks/1_data_clean.ipynb](notebooks/1_data_clean.ipynb), and the same steps run again in [src/preprocessing.py](src/preprocessing.py) when the models are trained.

### 1. Data sampling
For data sampling we took a sample of 50k reviews from the full Amazon Fine Food Reviews dataset on [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), which has 568,454 reviews in total. The notebook imports the data straight from Kaggle using `kagglehub`, so nobody has to download it themselves. The seed is set to `seed = 45`, so anyone in the group who runs the notebook gets the exact same 50k reviews. None of the CSV files are uploaded to the repo.

### 2. Labeling
For labelling we removed the 3-star reviews, which is around 3,728 reviews. These usually have mixed opinions, so they don't really fit as positive or negative. Reviews with 4 or 5 stars are marked positive (1) and reviews with 1 or 2 stars are marked negative (0). That left us with 46,272 reviews, around 84% positive and 16% negative.

### 3. Removing duplicates
We also removed duplicate reviews before working on the data. A duplicate here is the same user posting the same review at the same time on different product listings, so any rows with the same `UserId`, `Time` and `Text` were dropped. The number went down from 46,272 to 42,907 reviews. Doing this before the split keeps the same review from ending up in both training and testing, which would make our results look better than they really are.

### 4. Cleaning the text
Seven reviews had no Summary, so those were filled in with an empty string. A lot of reviews also had leftover HTML tags like `<br />`, and if we left them in, "br" would show up as one of the most common words, so we removed them. The Summary and Text were then joined into one column called `clean`, since the summary on its own can show the sentiment really clearly, like "Awful!" or "Best snack ever". We kept the stop words. The normal stop-word list includes "not", "no" and "never", and removing them would turn "not good" into "good".

### 5. Splitting the data
After cleaning, we did an 80/20 stratified split with seed 45, which gives 34,325 reviews for training and 8,582 for testing, both with 84.2 percent positive reviews. Stratified basically means both sets keep the same mix of positive and negative reviews.

The training code in `src/` takes another 10% out of the training set for validation, so the models can be tuned without touching the test data. That gives 30,892 reviews for training, 3,433 for validation, and the same 8,582 for testing.

Since most reviews are positive, a model that guesses "positive" every time would still get 84.2% accuracy. That is our baseline, and it's why we report F1 alongside accuracy.

### 6. Turning text into features
Models can't work with text directly, so each review has to be turned into numbers first.

We started with Bag of Words, where each review is turned into the number of times each word is used in it. It works on how often a word shows up, not on the order it appears in the sentence. We used `CountVectorizer` with the setting `min_df=5`, so only words that appear in at least 5 reviews are kept. That gets rid of typos and really rare words and makes it easier to focus on the important ones.

For Milestone 3 we tried different settings to make Naive Bayes better. These experiments are in the Milestone 3 cells of the notebook. Setting `ngram_range=(1,3)` lets the model also look at groups of two and three words, like "not good" or "did not work", so it can pick up negation. With `binary=True`, it only checks whether a word is in the review instead of counting it, so one word repeated many times can't take over the whole review. These settings were chosen with 5-fold cross-validation on the training data only, and the test set was used once at the very end. This raised Naive Bayes from 91.4% to 94.0% accuracy. The training script in `src/train.py` still uses the original settings for now.

The vectorizer is always fit on the training data first and then applied to the test data, so the model never sees the test reviews before it is tested.

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
