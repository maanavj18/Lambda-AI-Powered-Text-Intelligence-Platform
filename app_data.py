"""Dataset summary figures shown on the app's About tab.

The dataset CSVs are not committed, so the deployed app cannot compute these
live. Every value here is copied from the printed outputs of the Milestone 1
cells in notebooks/1_data_clean.ipynb. Model results are not kept here: the
app reads them from models/comparison_metrics.json, written by src/train.py.
"""

SAMPLE_ROWS = 50_000
FULL_DATASET_ROWS = 568_454

# Star rating -> number of reviews in the sample
SCORE_COUNTS = {1: 4_673, 2: 2_656, 3: 3_728, 4: 7_027, 5: 31_916}

# Sentiment labels after removing 3-star reviews: 1-2 stars negative, 4-5 stars positive
LABEL_COUNTS = {"Positive": 38_943, "Negative": 7_329}

AVG_WORDS = 79.0
MEDIAN_WORDS = 55

# Data quality issues found in the sample
DUPLICATE_REVIEWS = 3_625  # same user, time and text
MISSING_VALUES = {"Summary": 7, "ProfileName": 2}

# Reviews left after removing 3-star reviews and duplicates
CLEANED_ROWS = 42_907
