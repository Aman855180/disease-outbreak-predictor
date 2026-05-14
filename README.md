# Disease Outbreak Early Warning System

A machine learning system that detects disease outbreak risk from 
historical case trends — before the situation becomes uncontrollable.

[Live Demo](https://disease-outbreak-predictor-edcguujkj9xgstqnvcyz8a.streamlit.app/) · [Notebook](outbreak_eda.ipynb)

---

## The Problem

By the time a disease outbreak is obvious, containment is already 
difficult. Case counts are rising, healthcare systems are overwhelmed, 
and response teams are playing catch-up.

The goal of this project was simple: **can we detect the warning signs 
early enough to actually do something about it?**

---

## What I Built

An end-to-end ML system trained on WHO Ebola surveillance data 
(2014–2016) across 10 countries. It takes recent case count trends 
as input and outputs an outbreak risk probability — flagging 
dangerous situations before they peak.

The system is deployed as an interactive dashboard where a public 
health officer can enter case data for any region and immediately 
see the risk level, the historical trend, and exactly which factors 
are driving the prediction.

---

## How It Works

Raw case data is cumulative — so the first thing I had to do was 
derive weekly new cases by taking the difference per country over 
time. From there I engineered features that capture what actually 
precedes an outbreak:

- **Lag features** (cases 1, 2, 3 periods ago) — recent history
- **Rolling average & max** — what the trend looks like
- **Rate of change** — how fast cases are accelerating
- **Rolling std deviation** — volatility in recent weeks
- **Season & week of year** — temporal patterns

Defining the target variable was the most interesting challenge. 
I couldn't just use raw case counts — I had to decide what "outbreak" 
actually means in the data. I landed on:

> A week is an outbreak if new cases exceed 2× the 4-week rolling 
> average **and** the absolute count is above 20.

The second condition matters. Without it, the model flags 3 cases 
being 2× the average of 1.5 as an outbreak — which is noise, not signal.

---

## Model Selection

I trained three models and compared them on **recall** — not accuracy.

| Model | Recall | Precision | ROC-AUC |
|---|---|---|---|
| Decision Tree (tuned) | **0.906** | 0.35 | 0.933 |
| Decision Tree (baseline) | 0.875 | 0.45 | 0.925 |
| Logistic Regression | 0.750 | 0.13 | 0.919 |
| Random Forest | 0.469 | 0.88 | 0.982 |

Random Forest had the best AUC but missed 17 out of 32 real outbreaks. 
In a disease detection system, a missed outbreak is far more dangerous 
than a false alarm — so I chose the tuned Decision Tree which missed 
only 3.

Tuning with `RandomizedSearchCV` (scoring='recall') found that a 
shallow tree of `max_depth=3` generalized best. Deeper trees overfit 
the training outbreak patterns and failed on the test period.

---

## Explainability

I used SHAP to understand what the model actually learned.

The dominant feature by a large margin was **Rate_of_Change** — how 
fast cases are accelerating relative to the previous period. In one 
correctly detected outbreak, this single feature pushed the probability 
from 50% to 95.7%.

This makes epidemiological sense. It's not the absolute number of 
cases that signals danger — it's the acceleration.

---

## Results

- **Recall: 90.6%** — catches 29 out of 32 real outbreaks on test data
- **ROC-AUC: 0.933**
- Test period: June 2015 – March 2016 (the declining phase of the epidemic — the hardest period to evaluate on)

---

## Stack

Python · scikit-learn · pandas · SHAP · Streamlit · Plotly · joblib

---

## Project Structure
# disease-outbreak-predictor/
├── app.py                      # Streamlit dashboard
├── outbreak_eda.ipynb          # Full analysis — EDA, features, models, SHAP
├── outbreak_model.pkl          # Trained Decision Tree
├── scaler.pkl                  # Fitted StandardScaler
├── feature_cols.json           # Feature column names
└── ebola_2014_2016_clean.csv   # Cleaned WHO dataset
