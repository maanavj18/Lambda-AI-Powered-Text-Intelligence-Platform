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

The system has three layers. Raw reviews are converted into at least two numerical feature representations (for example, bag-of-words, TF-IDF, word embeddings, or character n-grams). 

two classical models (for example, Naive Bayes, logistic regression, a Gaussian mixture model, or k-nearest neighbors) are trained, tuned, and compared. At least one deep or modern NLP model (for example, a feedforward network, an LSTM, or a pretrained transformer such as DistilBERT) is trained and compared against those baselines.


