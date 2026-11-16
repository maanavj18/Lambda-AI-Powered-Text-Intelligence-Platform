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

 We use Naive Bayes and Logistic Regression our classical models to create a baseline. Then we train LSTM compare against those baselines.

## How to Run

Requires Python 3.10+. Run all commands from the repository root.

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
   pip install -r requirements.txt
   ```

3. **Get the trained model.** The app loads `sentiment.joblib` from `models/` (or `notebooks/models/`). If it is missing, run all cells in [notebooks/1_data_clean.ipynb](notebooks/1_data_clean.ipynb); the last cell trains the model and saves it.

4. **Launch the app**

   ```bash
   streamlit run app.py
   ```

   Streamlit opens the app in your browser (default http://localhost:8501). Type a review and click **Predict sentiment**.
