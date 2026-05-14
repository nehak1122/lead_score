# Lead Scoring Model

A machine learning model that predicts the likelihood of a sales lead converting into a paying customer. Built as part of the **TechnoHacks Internship Program - Data Science Track**.

## Overview

This project implements a complete ML pipeline for lead scoring, assigning each lead a numerical score (0-100) based on behavioral and demographic attributes. This enables sales teams to prioritize high-value leads and improve conversion rates.

## Dataset

- **Source:** [Kaggle Lead Scoring Dataset](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset)
- **Records:** 9,240 leads
- **Features:** 37 columns (mix of categorical and numerical)
- **Target:** Binary classification (Converted: 1/0)

## Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | 79.75% | 71.77% | 79.44% | **75.41%** | **86.94%** |
| XGBoost | 79.70% | 71.97% | 78.75% | 75.21% | 86.45% |
| Decision Tree | 78.43% | 69.65% | 79.44% | 74.23% | 84.88% |
| Logistic Regression | 78.43% | 71.16% | 75.38% | 73.21% | 85.67% |

### Best Model: Random Forest
- **F1-Score:** 0.7541
- **ROC-AUC:** 0.8694

### Top 5 Influential Features
1. Total Time Spent on Website (56.08%)
2. Current Occupation: Working Professional (9.80%)
3. Last Activity: SMS Sent (8.96%)
4. Current Occupation: Unemployed (5.93%)
5. Lead Origin: Lead Add Form (5.49%)

### Lead Priority Distribution
| Priority | Score Range | Count | Percentage |
|----------|-------------|-------|------------|
| Hot | ≥ 70 | 668 | 30.3% |
| Warm | 40-69 | 411 | 18.7% |
| Cold | < 40 | 1,123 | 51.0% |

## Project Structure

```
├── Lead_Scoring_Model.ipynb      # Complete Jupyter notebook
├── Lead_Scoring_Visualizations.pdf   # PDF report with all charts
├── Leads.csv                     # Dataset
├── lead.md                       # Product Requirements Document
├── lead_scores_output.csv        # Scored leads output
├── charts/                       # Individual visualization PNGs
│   ├── chart_01.png             # Missing Values Analysis
│   ├── chart_02.png             # Numerical Features (Before Outliers)
│   ├── chart_03.png             # Numerical Features (After Outliers)
│   ├── chart_04.png             # Target Distribution
│   ├── chart_05.png             # Conversion by Lead Source
│   ├── chart_06.png             # Conversion by Lead Origin
│   ├── chart_07.png             # Conversion by Last Activity
│   ├── chart_08.png             # Conversion by Occupation
│   ├── chart_09.png             # Features by Conversion (Histogram)
│   ├── chart_10.png             # Features by Conversion (Box Plot)
│   ├── chart_11.png             # Correlation Heatmap
│   ├── chart_12.png             # Model Comparison
│   ├── chart_13.png             # Confusion Matrices
│   ├── chart_14.png             # ROC Curves
│   ├── chart_15.png             # Feature Importance (RF)
│   ├── chart_16.png             # Feature Importance (XGBoost)
│   └── chart_17.png             # Lead Score Distribution
└── README.md
```

## Pipeline Steps

1. **Data Preprocessing**
   - Handle missing values (>40% threshold)
   - Replace "Select" placeholders with NaN
   - Remove identifier and leakage columns
   - Impute missing values (mode/median)
   - Outlier treatment using IQR capping

2. **Exploratory Data Analysis**
   - Target distribution analysis
   - Conversion rates by categorical features
   - Numerical feature distributions
   - Correlation heatmap

3. **Feature Engineering**
   - Label encoding for binary features
   - One-hot encoding for nominal features
   - Standard scaling for numerical features
   - RFE feature selection (25 features)

4. **Model Training**
   - Logistic Regression (baseline)
   - Decision Tree
   - Random Forest
   - XGBoost

5. **Lead Scoring**
   - Probability scores (0-100)
   - Priority buckets: Hot (≥70), Warm (40-69), Cold (<40)

## Technologies Used

- Python 3.9+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- XGBoost

## How to Run

```bash
# Clone the repository
git clone https://github.com/nehak1122/lead_score.git
cd lead_score

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# Open the notebook
jupyter notebook Lead_Scoring_Model.ipynb
```

## Author

**Neha Khetawat**

---

*TechnoHacks Internship Program - Data Science Track (Medium Level Task)*
