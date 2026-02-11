# High-Reliability System Failure Prediction

## Overview

This project implements a machine learning pipeline to predict system failures in high-reliability environments using multivariate sensor data.

The objective is to build and validate classification models that can detect potential failures early while minimizing false negatives in safety-critical scenarios.

---

## Problem Statement

Given multivariate sensor readings from a system, predict whether the system is likely to fail.

Binary classification:

* 0 -> Normal operation
* 1 -> Failure

Since failure detection is safety-sensitive, evaluation prioritizes recall and ROC-AUC rather than raw accuracy.

---

## Project Structure

```
high-reliability-failure-prediction/
├── data/
│   ├── generate_data.py
│   └── sensor_data.csv
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── app/
│   └── main.py
├── requirements.txt
└── README.md
```

---

## Methodology

1. Data Generation / Loading
   Synthetic multivariate sensor data simulates system degradation and class imbalance.

2. Preprocessing

    * Stratified train-test split
    * Feature scaling
    * Class imbalance handling

3. Models Implemented

    * Logistic Regression
    * Random Forest
    * XGBoost

4. Model Validation

    * Stratified K-Fold Cross-Validation
    * ROC-AUC Score
    * F1-Score
    * Confusion Matrix

Evaluation emphasizes minimizing false negatives in safety-critical contexts.

---

## Handling Class Imbalance

Failure cases are less frequent than normal cases.

Strategies applied:

* class_weight="balanced" for linear and tree models
* scale_pos_weight for XGBoost
* Stratified data splitting

---

## Results

Models are compared using:

* Cross-validation ROC-AUC
* Test ROC-AUC
* F1-score

Tree-based models typically perform better when nonlinear sensor interactions exist.

---

## How to Run

1. Install dependencies

`pip install -r requirements.txt`

2. Generate dataset

`cd data
python generate_data.py`

3. Train model

`cd src
python train.py`

4. Evaluate model

`python evaluate.py`

5. Run API (optional)

`uvicorn app.main:app --reload`

Open in browser:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Tech Stack

Python
Scikit-learn
XGBoost
Pandas
NumPy
FastAPI