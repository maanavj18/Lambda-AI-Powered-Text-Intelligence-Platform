# Data

We don't upload the CSV files to GitHub because they're too large, and the project guidelines ask us not to commit them. Instead, anyone on the team can recreate the exact same files by running our notebook.

## Where the data comes from

Our project uses the Amazon Fine Food Reviews dataset from Kaggle, collected by the Stanford Network Analysis Project. The full dataset has 568,454 reviews written between October 1999 and October 2012. You can find it here: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

You don't need to download it yourself. The notebook pulls it in automatically using kagglehub.

## How to recreate our files

Open `notebooks/1_data_clean.ipynb` in Google Colab and run all the cells from top to bottom. It takes about two minutes and creates three files:

- `sample_50k.csv`: a random sample of 50,000 reviews. We use a fixed seed (42) so everyone gets the same rows.
- `train.csv`: 34,236 reviews used for training (80%)
- `test.csv`: 8,559 reviews used for testing (20%)

## What we did to clean the data

We removed the 3-star reviews because they usually have mixed opinions and don't clearly fit as positive or negative. Reviews with 4 or 5 stars are labeled positive (1) and reviews with 1 or 2 stars are labeled negative (0).

We also removed duplicate reviews, which are the same user posting the same text at the same time across different product listings. We did this before splitting so the same review can't end up in both the training and test sets. After that we stripped out leftover HTML tags like `<br />` and combined the Summary and Text fields into one column called `clean`.

The split is stratified, so both the training and test sets keep the same balance of about 85% positive and 15% negative reviews.
