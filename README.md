# Hotel Booking Cancellation Prediction

A machine learning pipeline that predicts hotel booking cancellations using the [Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) from Kaggle.

## Overview

This project builds and evaluates three classification models — Logistic Regression, Random Forest, and Decision Tree — to predict whether a hotel booking will be cancelled. It emphasizes rigorous **data leakage prevention** throughout, with the train/test split occurring before any EDA or preprocessing is fitted.

## Features

- Full EDA with themed visualizations (target distribution, correlations, lead time, ADR, seasonality, skewness)
- PCA as a diagnostic tool (variance structure and feature loadings — not used for final prediction)
- Three models trained with `GridSearchCV` and `StratifiedKFold` cross-validation
- Decision threshold tuning on the best model to optimize F1 / Recall trade-off
- Business-oriented insights at each analytical step

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
kagglehub
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn kagglehub
```

## Usage

```bash
python HotelBookingProject_v3.py
```

The script downloads the dataset automatically via `kagglehub`. A Kaggle account and API token are required.

## Pipeline Summary

| Step | Description |
|------|-------------|
| 1 | Load data via `kagglehub` |
| 2 | Drop target-leakage columns (`reservation_status`, `reservation_status_date`) |
| 3 | Clean data (duplicates, missing values, domain fixes) |
| 4-5 | Train/test split (80/20, stratified) — **before EDA** |
| 5a-5g | EDA on training data only (distributions, correlations, seasonality) |
| 6-7 | Feature selection: drop numeric features with \|r\| < 0.02 |
| 8 | `ColumnTransformer`: OHE (low-cardinality), OrdinalEncoder (high-cardinality), StandardScaler (numeric) |
| 9 | PCA diagnostic (scree plot, 2D scatter, loadings heatmap) |
| 10 | `GridSearchCV` training: Logistic Regression, Random Forest, Decision Tree |
| 11 | Test-set evaluation: AUC-ROC, F1, confusion matrix, ROC curves |
| 12 | Threshold tuning on Random Forest (sweep 0.10–0.90, optimize F1 then minimize Type II error) |

## Leakage Prevention

Leakage is addressed at every stage:

- `reservation_status*` columns dropped from raw data before any processing
- Train/test split occurs **before** EDA, correlation analysis, and feature selection
- `OHE_THRESHOLD` cardinality is computed from `X_train` only
- All transformers (imputer, scaler, encoder) are fitted inside `Pipeline` / `GridSearchCV` on training folds only
- Safety assertions verify leakage columns are absent at four checkpoints

## Output Figures

| File | Description |
|------|-------------|
| `01_target_distribution.png` | Class balance (pie + bar) |
| `02_correlation_analysis.png` | Feature correlation heatmap + bar chart |
| `03_lead_time.png` | Lead time boxplot + cancellation rate by bucket |
| `04_price_adr.png` | ADR distribution + cancellation rate by price bucket |
| `05_cancel_by_*.png` | Cancellation rate by market segment, customer type, deposit type |
| `06_seasonality.png` | Monthly bookings and cancellation rate |
| `07_boxplot_top5_skewed.png` | Boxplots of 5 most skewed features |
| `08_boxplot_all_numeric.png` | Boxplots for all numeric features |
| `PCA_A_explained_variance.png` | Scree plot and cumulative variance |
| `PCA_B_2D_scatter.png` | 2D PCA class separability |
| `PCA_C_loadings_heatmap.png` | PCA loadings for top features |
| `09_model_comparison.png` | CV AUC vs Test AUC vs F1 by model |
| `09b_type1_type2_error_schema.png` | Type I and Type II error rates by model |
| `10_roc_curves.png` | ROC curves for all three models |
| `11_confusion_matrix.png` | Confusion matrix for the best model |
| `12_feature_importance.png` | Top 15 feature importances |
| `13a_threshold_f1_recall.png` | F1 and Recall vs decision threshold |
| `13b_threshold_errors.png` | Type I and Type II error vs threshold |
| `13c_threshold_overview.png` | All threshold metrics combined |
| `13d_confusion_matrix_tuned.png` | Confusion matrix at default vs optimal threshold |

## Key Business Insights

- **Lead time**: Bookings made >90 days ahead cancel at >50%. Recommend requiring deposits for long-lead bookings.
- **Deposit type**: Non-refundable bookings have near-zero cancellation rates.
- **Market segment**: Online travel agents and group bookings have the highest cancellation rates.
- **Threshold tuning**: Lowering the decision threshold below 0.5 catches more true cancellations (lower Type II error) at the cost of slightly more false alarms — typically the right trade-off for hotel revenue management.

## Model Notes

- **PCA-LR was excluded** from the final model comparison. PCA applied to a mixed feature space (scaled numeric + OHE + ordinal-encoded high-cardinality columns) distorts principal components because ordinal codes are arbitrary integers. PCA is kept in Section 9 for diagnostics only.
- **Decision Tree overfitting** is explicitly measured: Train AUC vs CV AUC vs Test AUC are printed so the generalization gap is visible.
- **Random Forest** is used for threshold tuning as it produces the best-calibrated probability estimates among the three models.
  
## Report
- a well detailed report explaining the whole project.
