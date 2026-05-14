# Diabetes Risk Prediction (Healthcare ML Project)

## Overview

Built an end-to-end machine learning pipeline to predict diabetes risk using large-scale healthcare data. The project focuses on feature engineering, handling imbalanced data, and improving model stability for real-world applicability.

---

## Dataset

* Size: 700K+ healthcare records
* Features include:

  * Glucose levels
  * Blood pressure
  * BMI
  * Cholesterol metrics
  * Lifestyle indicators

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost, LightGBM
* Matplotlib, Seaborn

---

## Project Workflow

1. Data Cleaning & Preprocessing
2. Feature Engineering (clinical + statistical features)
3. Handling Class Imbalance
4. Model Training & Optimization
5. Model Evaluation & Validation

---

## Feature Engineering

* Created BMI-based risk flags
* Generated blood pressure bands
* Derived cholesterol ratios
* Designed lifestyle-based risk scores

---

## Models Used

* XGBoost
* LightGBM

---

## Results

* ROC-AUC Score: **0.70**
* Improved model stability using:

  * Feature pruning
  * Validation-based tuning
  * Seed averaging

---

## Key Insights

* Feature engineering significantly improved prediction performance
* Handling class imbalance was critical for better recall
* Ensemble boosting models performed better than baseline models

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

---

## Project Structure

* notebooks/ → model development
* data/ → dataset
* src/ → scripts (if any)

---

## Future Improvements

* Hyperparameter tuning using Optuna
* Deep learning models for comparison
* Deployment using Streamlit

---

## Author

Parv Gupta
