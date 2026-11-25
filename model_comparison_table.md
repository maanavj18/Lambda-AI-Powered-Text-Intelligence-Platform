# Model Comparison

All models were evaluated on the same 8,582 held-out Amazon Fine Food Reviews.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Training time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority-class baseline | 0.8418 | 0.4209 | 0.5000 | 0.4570 | 0.0019 |
| Multinomial Naive Bayes | 0.9142 | 0.8343 | 0.8540 | 0.8437 | 0.0107 |
| Logistic Regression | 0.9322 | 0.8877 | 0.8476 | 0.8659 | 2.3468 |
| LSTM | **0.9406** | **0.8967** | **0.8750** | **0.8854** | 316.2166 |

The LSTM had the highest macro F1, 1.95 percentage points above Logistic Regression. [Evaluation setup and confusion matrices](ml_engineer_model_comparison.md) · [Full-precision results](models/comparison_metrics.json)
