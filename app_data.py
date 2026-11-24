"""Numbers displayed by the Streamlit app.

The dataset CSVs are not committed, so the deployed app cannot compute these
live. Every value here is copied from the printed outputs of
notebooks/1_data_clean.ipynb. When the notebook is re-run or new results are
reported, update this file and the app picks the changes up.
"""

# --- Dataset summary (Milestone 1), from the 50,000-review sample -----------
SAMPLE_ROWS = 50_000
FULL_DATASET_ROWS = 568_454

# Star rating -> number of reviews
SCORE_COUNTS = {1: 4_673, 2: 2_656, 3: 3_728, 4: 7_027, 5: 31_916}

# Sentiment labels: 1-2 stars = negative, 4-5 stars = positive, 3 stars left out
LABEL_COUNTS = {"Positive": 38_943, "Negative": 7_329}

AVG_WORDS = 79.0
MEDIAN_WORDS = 55

# Data quality issues found in the sample
DUPLICATE_REVIEWS = 3_625  # same user, time and text
MISSING_VALUES = {"Summary": 7, "ProfileName": 2}

# Rows remaining after each cleaning step
CLEANING_STEPS = {
    "Sample": 50_000,
    "3-star reviews removed": 46_272,
    "Duplicates removed": 42_907,
}

# --- Train / test split (80% / 20%, stratified by label) ---------------------
# Test counts match the support column of the notebook's classification report;
# training counts are the label counts in notebooks/train.csv.
SPLIT = {
    "Training set": {"Positive": 28_894, "Negative": 5_431},
    "Test set": {"Positive": 7_224, "Negative": 1_358},
}

# --- Model results on the held-out test set ----------------------------------
# Precision, recall and F1 are macro averages.
# Confusion matrix layout: [[true negative, false positive], [false negative, true positive]]
# To add a model, add an entry here; the Models tab draws it automatically.
MODEL_RESULTS = {
    "Majority baseline": {
        "accuracy": 0.8418,
        "precision": 0.4209,
        "recall": 0.5000,
        "f1": 0.4570,
        "confusion_matrix": [[0, 1_358], [0, 7_224]],
    },
    "Multinomial Naive Bayes": {
        "accuracy": 0.9142,
        "precision": 0.8341,
        "recall": 0.8549,
        "f1": 0.8439,
        "confusion_matrix": [[1_043, 315], [421, 6_803]],
    },
}
