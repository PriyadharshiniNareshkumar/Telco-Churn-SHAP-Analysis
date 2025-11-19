# Telco Customer Churn Prediction with SHAP Analysis

## Project Overview
This project develops an XGBoost classification model for predicting telecom customer churn, 
with rigorous interpretation using SHAP (SHapley Additive exPlanations) for explainability.

## Key Metrics
- **AUC**: 0.815
- **F1-Score**: 0.595
- **Recall**: 0.684

## Features
- Data preprocessing and class imbalance handling
- XGBoost model training and tuning
- Global SHAP feature importance analysis
- Local SHAP force plots for 3 distinct customer profiles
- Cohort-based comparative analysis
- 3 strategic recommendations for churn reduction

## Files
- `Telco_Churn_SHAP_Analysis.ipynb` - Complete Jupyter notebook with code and analysis
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Original dataset
- `SHAP_Feature_Importance.csv` - Generated feature importance scores

## Results
- **Predicted Churn Reduction**: 12-17%
- **Expected Revenue Protection**: $3.6M - $5.4M annually

## Recommendations
1. Early engagement for new customers (Month 1-3)
2. Contract conversion program targeting month-to-month customers
3. Predictive add-on service cross-sell program

## How to Run
1. Install dependencies: `pip install pandas numpy scikit-learn xgboost shap`
2. Open `Telco_Churn_SHAP_Analysis.ipynb` in Jupyter
3. Run all cells to reproduce analysis

## Author
priyadharshini N 
Date: November 19, 2025

