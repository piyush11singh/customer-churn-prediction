# Customer Churn Prediction (Banking)

Predicting which bank customers are likely to churn (close their account / leave), using account activity, demographics, and product-holding data — with the goal of enabling targeted, cost-efficient retention strategies.

## Problem Statement

Customer attrition directly impacts recurring revenue in retail banking. Rather than applying blanket retention efforts across the entire customer base, this project builds a model to identify **which customers are at risk of churning and why**, so retention resources can be prioritized toward the highest-impact segments.

## Dataset

- **Source:** Bank Customer Churn Prediction dataset (Kaggle)
- **Size:** 10,000 customer records, 12 columns, no missing values
- **Target variable:** `churn` (1 = customer left, 0 = customer stayed)
- **Class balance:** ~20.4% churn rate

**Features:** `credit_score`, `country`, `gender`, `age`, `tenure`, `balance`, `products_number`, `credit_card`, `active_member`, `estimated_salary`

## Approach

1. **Exploratory Data Analysis**
   - Distribution plots (KDE, violin) for all numeric features split by churn status
   - Churn rate breakdowns by `products_number`, `active_member`, `country`, `gender`
   - Correlation heatmap across numeric features

2. **Feature Engineering**
   - `balance_per_product` — balance normalized by number of products held
   - `salary_balance_ratio` — financial health proxy
   - `age_group` — binned age brackets
   - `tenure_bucket` — binned tenure brackets
   - `high_balance` — flag for customers above the 75th percentile balance

3. **Modeling**
   - Built a `scikit-learn` pipeline combining preprocessing (imputation, scaling, one-hot encoding) with classification, to avoid data leakage and keep training/inference consistent
   - Compared 5 models via 5-fold stratified cross-validation, scored on ROC-AUC:

     | Model | CV ROC-AUC |
     |---|---|
     | Logistic Regression | 0.788 |
     | Random Forest | 0.849 |
     | **Gradient Boosting** | **0.863** |
     | AdaBoost | 0.846 |
     | SVC | 0.835 |

   - **Gradient Boosting** selected as the best-performing model

4. **Evaluation (held-out test set)**

   | Metric | Score |
   |---|---|
   | Accuracy | 86.8% |
   | Precision | 78.0% |
   | Recall | 48.9% |
   | F1-score | 60.1% |
   | ROC-AUC | 0.869 |

5. **Explainability**
   - Extracted feature importances from the trained Gradient Boosting model to validate and quantify the EDA insights

   | Feature | Importance |
   |---|---|
   | age | 32.8% |
   | products_number | 26.6% |
   | balance_per_product | 6.3% |
   | balance | 5.7% |
   | active_member | ~10% (combined) |
   | country_Germany | 5.1% |

## Key Findings

- **Age and product holdings are the dominant churn drivers**, together explaining ~58% of the model's predictive signal
- **Single-product customers churn at a materially higher rate** than those holding 2+ products
- **Inactive members** are meaningfully more likely to churn than active ones
- **Germany shows a distinctly higher churn signal** than France or Spain — worth a follow-up qualitative investigation

## Business Recommendations

1. **Cross-sell campaigns targeted at single-product customers** — the highest-leverage retention lever identified
2. **Age-segmented retention messaging** rather than a one-size-fits-all approach, since age is the strongest single predictor
3. **Proactive re-engagement for inactive members** via check-ins or personalized offers
4. **Regional investigation into the Germany churn signal** — likely a service, pricing, or competitive issue worth qualitative follow-up
5. **Threshold tuning before deployment** — current recall (49%) means roughly half of churners go undetected; given that losing a customer is typically costlier than an unnecessary retention offer, a lower classification threshold should be evaluated to trade some precision for higher recall

## Repository Structure

```
customer-churn-prediction/
├── README.md
├── data/
│   └── Bank_Customer_Churn_Prediction.csv
├── analysis.ipynb
└── requirements.txt
```

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `joblib`

## Next Steps

- Threshold tuning / precision-recall curve analysis to address recall gap
- Refactor feature engineering into a shared `preprocess_new_data()` function for consistent training/inference
- Deploy as an interactive Streamlit app for live predictions
