"""Recreate the review sample and held-out split from the project notebook."""

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 45
DATASET = "snap/amazon-fine-food-reviews/versions/2"
SAMPLE_SIZE = 50_000
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.1
HTML_TAG = re.compile(r"<.*?>")


def clean_review(text: str, summary: str = "") -> str:
    return HTML_TAG.sub(" ", f"{summary} {text}")


def load_splits(reviews_path=None):
    if reviews_path is None:
        reviews_path = Path(__file__).resolve().parents[1] / "data" / "Reviews.csv"
        if not reviews_path.is_file():
            import kagglehub

            reviews_path = kagglehub.dataset_download(DATASET, path="Reviews.csv")

    columns = ["Id", "UserId", "Time", "Score", "Summary", "Text"]
    sample = pd.read_csv(reviews_path, usecols=columns).sample(
        SAMPLE_SIZE, random_state=SEED
    ).reset_index(drop=True)
    reviews = sample[sample.Score != 3].drop_duplicates(["UserId", "Time", "Text"]).copy()
    reviews["label"] = (reviews.Score >= 4).astype(int)
    combined = reviews.Summary.fillna("") + " " + reviews.Text.fillna("")
    reviews["clean"] = combined.str.replace(HTML_TAG, " ", regex=True)
    development, test = train_test_split(
        reviews, test_size=TEST_FRACTION, stratify=reviews.label, random_state=SEED
    )
    train, validation = train_test_split(
        development,
        test_size=VALIDATION_FRACTION,
        stratify=development.label,
        random_state=SEED,
    )
    return train, validation, test
