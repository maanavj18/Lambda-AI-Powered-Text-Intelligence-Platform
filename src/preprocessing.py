"""Recreate the review sample and held-out split from the project notebook.

Run `python -m src.preprocessing` to download Reviews.csv and write the cleaned sample.
"""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 45
DATASET = "snap/amazon-fine-food-reviews/versions/2"
SAMPLE_SIZE = 50_000
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.1
HTML_TAG = re.compile(r"<.*?>")
COLUMNS = ["Id", "UserId", "Time", "Score", "Summary", "Text"]


def data_dir():
    return Path(__file__).resolve().parents[1] / "data"


def clean_review(text: str, summary: str = "") -> str:
    return HTML_TAG.sub(" ", f"{summary} {text}")


def download_reviews(destination=None):
    """Return Reviews.csv, copying Kaggle dataset version 2 into data/ when it is missing."""
    destination = Path(destination) if destination is not None else data_dir() / "Reviews.csv"
    if destination.is_file():
        return destination
    import kagglehub

    downloaded = Path(kagglehub.dataset_download(DATASET, path="Reviews.csv"))
    if downloaded.resolve() == destination.resolve():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, destination)
    return destination


def prepare_reviews(reviews_path):
    """Sample, drop 3-star and duplicate reviews, and build the label and cleaned text."""
    sample = pd.read_csv(reviews_path, usecols=COLUMNS).sample(
        SAMPLE_SIZE, random_state=SEED
    ).reset_index(drop=True)
    reviews = sample[sample.Score != 3].drop_duplicates(["UserId", "Time", "Text"]).copy()
    reviews["Summary"] = reviews.Summary.fillna("")
    reviews["Text"] = reviews.Text.fillna("")
    reviews["label"] = (reviews.Score >= 4).astype(int)
    reviews["clean"] = (reviews.Summary + " " + reviews.Text).str.replace(HTML_TAG, " ", regex=True)
    return reviews


def load_splits(reviews_path=None):
    if reviews_path is None:
        reviews_path = download_reviews()
    reviews = prepare_reviews(reviews_path)
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


def main():
    parser = argparse.ArgumentParser(
        description="Download Reviews.csv and write the cleaned review sample."
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        help="Path to Kaggle's Reviews.csv. Downloads into data/ when omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Cleaned CSV to write. Defaults to data/reviews_cleaned.csv.",
    )
    args = parser.parse_args()
    reviews_path = download_reviews() if args.reviews is None else args.reviews
    output = args.output or (data_dir() / "reviews_cleaned.csv")
    reviews = prepare_reviews(reviews_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(output, index=False)
    print(f"Reviews: {reviews_path}", flush=True)
    print(f"Cleaned: {output} ({len(reviews):,} rows)", flush=True)


if __name__ == "__main__":
    main()
