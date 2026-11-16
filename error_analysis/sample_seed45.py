"""Seed-45 group sample + keyword mismatch analysis.

1. Draws the 50,000-review group sample (seed 45) and saves it to Downloads.
2. Finds mismatched reviews: 5-star reviews that use strongly negative words,
   and 1-star reviews that use strongly positive words. Simple keyword check,
   so expect some false hits (e.g. "not bad", "I used to love this").
3. Prints a summary, the most common matched words and example reviews, and
   saves all mismatches to error_analysis/mismatched_reviews_seed45.csv.

Usage:
    python error_analysis/sample_seed45.py      # 5 examples of each kind
    python error_analysis/sample_seed45.py 20   # 20 examples of each kind
"""
import os
import re
import sys
import textwrap
from collections import Counter

import pandas as pd, matplotlib.pyplot as plt

reviews_path = "Reviews.csv"
seed = 45
n_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 5

# --- Group sample -------------------------------------------------------------
df = pd.read_csv(reviews_path)
sample = df.sample(n=50000, random_state=seed)

out_path = os.path.join(os.path.expanduser("~"), "Downloads", f"group_sample_seed{seed}.csv")
sample.to_csv(out_path, index=False)

print(f"Saved {len(sample)} rows to {out_path}")
print(sample[['Score', 'Text']].head(3))

# --- Find mismatched reviews ----------------------------------------------------
neg_words = ['terrible', 'worst', 'yuck', 'awful', 'disappointing', 'tasteless', 'worse', 'horrible', 'disappointed', 'rancid']
pos_words = ['beat', 'wonderful', 'amazing', 'perfect', 'excellent', 'delicious', 'yum', 'awesome', 'pleasantly', 'yummy']


def find_words(text, vocab):
    found = re.findall(r"\b(" + "|".join(vocab) + r")\b", text.lower())
    return sorted(set(found))


sample = sample.copy()
sample["clean"] = (sample.Summary.fillna("") + " " + sample.Text).str.replace(r"<.*?>", " ", regex=True)
sample["neg_found"] = sample.clean.apply(lambda t: find_words(t, neg_words))
sample["pos_found"] = sample.clean.apply(lambda t: find_words(t, pos_words))

five_star_neg = sample[(sample.Score == 5) & (sample.neg_found.str.len() > 0)]
one_star_pos = sample[(sample.Score == 1) & (sample.pos_found.str.len() > 0)]

print(f"\n5-star reviews with negative words: {len(five_star_neg)} of {(sample.Score == 5).sum()}")
print(f"1-star reviews with positive words: {len(one_star_pos)} of {(sample.Score == 1).sum()}")

neg_counts = Counter(w for words in five_star_neg.neg_found for w in words)
pos_counts = Counter(w for words in one_star_pos.pos_found for w in words)
print("\nMost common negative words in 5-star reviews:", neg_counts.most_common(5))
print("Most common positive words in 1-star reviews:", pos_counts.most_common(5))

# --- Examples -------------------------------------------------------------------
def show(rows, words_col, title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    for _, r in rows.head(n_examples).iterrows():
        print(f"\n[{r.ProductId}] {r.Score} stars | matched: {', '.join(r[words_col])}")
        print(f"Summary: {r.Summary}")
        text = re.sub(r"<.*?>", " ", r.Text)
        print(textwrap.fill(f"Text: {textwrap.shorten(text, width=600, placeholder=' ...')}", width=80))


show(five_star_neg, "neg_found", "5 STARS, NEGATIVE WORDS")
show(one_star_pos, "pos_found", "1 STAR, POSITIVE WORDS")

# --- Save -----------------------------------------------------------------------
cols = ["ProductId", "Score", "Summary"]
mismatched = pd.concat([five_star_neg, one_star_pos])
error_dir = os.path.dirname(os.path.abspath(__file__))  # this error_analysis folder
mismatched_path = os.path.join(error_dir, f"mismatched_reviews_seed{seed}.csv")
mismatched[cols + ["Text", "neg_found", "pos_found"]].to_csv(mismatched_path, index=False)
print(f"\nSaved {len(mismatched)} mismatched reviews to {mismatched_path}")
