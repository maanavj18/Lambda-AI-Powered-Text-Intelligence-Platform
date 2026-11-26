"""Train and compare sentiment models with `python -m src.train`."""

import argparse
import json
import platform
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import torch
from joblib import dump
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluate import measure, save_loss_curve
from .lstm import SentimentLSTM, build_vocabulary, encode_reviews
from .preprocessing import SEED, load_splits


BATCH_SIZE = 128
LEARNING_RATE = 0.003
PATIENCE = 3


def fit_classical(train, validation, test, output):
    vectorizer = CountVectorizer(min_df=5)
    train_counts = vectorizer.fit_transform(train.clean)
    validation_counts = vectorizer.transform(validation.clean)
    test_counts = vectorizer.transform(test.clean)
    candidates = [
        (
            "Multinomial Naive Bayes",
            "naive_bayes",
            "alpha",
            [(alpha, MultinomialNB(alpha=alpha)) for alpha in (0.1, 0.5, 1.0, 2.0, 5.0)],
        ),
        (
            "Logistic Regression",
            "logistic_regression",
            "C",
            [(c, LogisticRegression(C=c, max_iter=1000)) for c in (0.01, 0.05, 0.1, 0.5, 1.0, 5.0)],
        ),
    ]

    baseline = DummyClassifier(strategy="most_frequent")
    start = perf_counter()
    baseline.fit(train_counts, train.label)
    elapsed = perf_counter() - start
    rows = {
        "Majority-class baseline": {
            **measure(test.label, baseline.predict(test_counts)),
            "training_seconds": elapsed,
        }
    }

    for name, filename, parameter, models in candidates:
        trials = []
        for value, model in models:
            start = perf_counter()
            model.fit(train_counts, train.label)
            elapsed = perf_counter() - start
            validation_f1 = measure(validation.label, model.predict(validation_counts))["macro_f1"]
            trials.append((validation_f1, value, elapsed, model))
        _, value, elapsed, model = max(trials, key=lambda trial: trial[0])
        score = measure(test.label, model.predict(test_counts))
        rows[name] = {
            **score,
            "training_seconds": elapsed,
            "parameters": {parameter: value},
            "validation_trials": [
                {parameter: trial_value, "macro_f1": f1} for f1, trial_value, _, _ in trials
            ],
        }
        pipeline = Pipeline([("vectorizer", vectorizer), ("classifier", model)])
        dump(pipeline, output / f"{filename}.joblib")
        print(f"{name}: {score['macro_f1']:.4f} test macro F1", flush=True)

    return rows, len(vectorizer.vocabulary_)


def fit_improved_naive_bayes(train, validation, test, output, threads):
    """Fit the notebook's phrase-level Naive Bayes and export it separately.

    The search matches the Milestone 3 grid: binary counts, phrases of up to three
    words, and 5-fold macro F1 on the notebook's training split (this script's
    training and validation reviews together). The test split is scored once.
    """
    development = pd.concat([train, validation]).sort_index()
    search = GridSearchCV(
        Pipeline([
            ("vectorizer", CountVectorizer(min_df=2, binary=True)),
            ("classifier", MultinomialNB()),
        ]),
        {
            "vectorizer__ngram_range": [(1, 2), (1, 3)],
            "classifier__alpha": [0.1, 0.2, 0.3, 0.5, 1.0],
        },
        cv=5,
        scoring="f1_macro",
        n_jobs=threads,
        refit=False,
        verbose=1,
    )
    search.fit(development.clean, development.label)
    best = search.best_params_
    pipeline = Pipeline([
        (
            "vectorizer",
            CountVectorizer(
                min_df=2, binary=True, ngram_range=best["vectorizer__ngram_range"]
            ),
        ),
        ("classifier", MultinomialNB(alpha=best["classifier__alpha"])),
    ])
    start = perf_counter()
    pipeline.fit(development.clean, development.label)
    elapsed = perf_counter() - start
    score = measure(test.label, pipeline.predict(test.clean))
    result = {
        **score,
        "training_seconds": elapsed,
        "vocabulary_size": len(pipeline.named_steps["vectorizer"].vocabulary_),
        "parameters": {
            "min_df": 2,
            "binary": True,
            "ngram_range": list(best["vectorizer__ngram_range"]),
            "alpha": best["classifier__alpha"],
        },
        "cross_validation_macro_f1": float(search.best_score_),
        "cross_validation_trials": [
            {
                "ngram_range": list(params["vectorizer__ngram_range"]),
                "alpha": params["classifier__alpha"],
                "macro_f1": float(mean_f1),
            }
            for params, mean_f1 in zip(
                search.cv_results_["params"], search.cv_results_["mean_test_score"]
            )
        ],
    }
    dump(pipeline, output / "naive_bayes_improved.joblib")
    print(f"Improved Naive Bayes: {score['macro_f1']:.4f} test macro F1", flush=True)
    return result


def lstm_loader(reviews, vocabulary, shuffle=False):
    tokens = encode_reviews(reviews.clean, vocabulary)
    labels = torch.tensor(reviews.label.to_numpy(), dtype=torch.float32)
    return DataLoader(
        TensorDataset(tokens, labels),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(SEED),
    )


def lstm_epoch(model, batches, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    outputs = []
    with torch.set_grad_enabled(training):
        for token_ids, labels in batches:
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            else:
                outputs.append(logits)
            total_loss += loss.item() * len(labels)
    predictions = None if training else (torch.cat(outputs) >= 0).numpy().astype(int)
    return total_loss / len(batches.dataset), predictions


def fit_lstm(train, validation, test, output, epochs):
    vocabulary = build_vocabulary(train.clean)
    train_batches = lstm_loader(train, vocabulary, shuffle=True)
    validation_batches = lstm_loader(validation, vocabulary)
    model = SentimentLSTM(len(vocabulary) + 2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    best_f1 = -1.0
    best_epoch = 0
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        training_loss, _ = lstm_epoch(model, train_batches, criterion, optimizer)
        validation_loss, predictions = lstm_epoch(model, validation_batches, criterion)
        validation_f1 = measure(validation.label, predictions)["macro_f1"]
        history.append({
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_macro_f1": validation_f1,
        })
        print(
            f"LSTM epoch {epoch}: train loss {training_loss:.4f}, "
            f"val loss {validation_loss:.4f}, val macro F1 {validation_f1:.4f}",
            flush=True,
        )
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        if epoch - best_epoch >= PATIENCE:
            break
    elapsed = perf_counter() - start

    model.load_state_dict(best_state)
    _, predictions = lstm_epoch(model, lstm_loader(test, vocabulary), criterion)
    result = {
        **measure(test.label, predictions),
        "training_seconds": elapsed,
        "vocabulary_size": len(vocabulary),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selected_epoch": best_epoch,
        "parameters": {"batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "patience": PATIENCE},
        "history": history,
    }
    torch.save({"vocabulary": vocabulary, "state_dict": best_state}, output / "lstm.pt")
    save_loss_curve(history, best_epoch, output / "lstm_loss.png")
    print(f"LSTM: {result['macro_f1']:.4f} test macro F1", flush=True)
    return result


def comparison_metrics(rows, splits, bow_vocabulary_size, threads):
    return {
        "seed": SEED,
        "hardware": {"processor": platform.processor(), "cpu_threads": threads},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "splits": {
            name: {
                "negative": int(split.label.eq(0).sum()),
                "positive": int(split.label.eq(1).sum()),
                "ordered_id_sha256": sha256(",".join(split.Id.astype(str)).encode("ascii")).hexdigest(),
            }
            for name, split in splits.items()
        },
        "bag_of_words_vocabulary_size": bow_vocabulary_size,
        "models": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, help="Path to Kaggle's Reviews.csv")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("--epochs and --threads must be positive")

    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    train, validation, test = load_splits(args.reviews)
    splits = {"training": train, "validation": validation, "test": test}
    print(", ".join(f"{name}: {len(split):,}" for name, split in splits.items()), flush=True)

    output = args.models_dir
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".comparison-", dir=output) as temporary:
        staging = Path(temporary)
        with threadpool_limits(limits=args.threads):
            rows, bow_vocabulary_size = fit_classical(train, validation, test, staging)
            rows["Improved Naive Bayes"] = fit_improved_naive_bayes(
                train, validation, test, staging, args.threads
            )
            rows["LSTM"] = fit_lstm(train, validation, test, staging, args.epochs)

        results = comparison_metrics(rows, splits, bow_vocabulary_size, args.threads)
        metrics = staging / "comparison_metrics.json"
        metrics.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        for artifact in staging.iterdir():
            if artifact != metrics:
                artifact.replace(output / artifact.name)
        metrics.replace(output / metrics.name)


if __name__ == "__main__":
    main()
