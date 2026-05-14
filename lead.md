Here's the full PRD in copy-paste format:

---

# PRODUCT REQUIREMENTS DOCUMENT
# Lead Scoring Model — Feature-Based Conversion Prediction

| Field | Details |
|---|---|
| Project Name | Lead Scoring Model (Feature-Based Prediction) |
| Difficulty Level | Medium |
| Program | TechnoHacks Internship Program — Data Science Track |
| Document Version | 1.0 |
| Date | May 13, 2026 |
| Status | Draft — Ready for Implementation |

---

## 1. Executive Summary

This document outlines the product requirements for building a Lead Scoring Model that predicts the likelihood of a sales lead converting into a paying customer. The model will assign each lead a numerical score (0–100) based on behavioral and demographic attributes, enabling the sales team to prioritize high-value leads and improve overall conversion rates.

The project is part of the TechnoHacks Data Science Internship Program (Medium Level Task) and will deliver a complete machine learning pipeline — from data preprocessing through model comparison — implemented in Python using Scikit-learn and documented in a Jupyter/Colab notebook.

---

## 2. Problem Statement

Sales teams in lead-driven businesses (EdTech, SaaS, real estate, finance) routinely receive thousands of inbound leads, but only a small fraction (~10–15%) actually convert. Without a systematic way to rank these leads, sales reps waste time on low-intent prospects while letting hot leads go cold.

**Business Question:**
"Given a new lead's profile and behavior, how likely are they to convert — and which leads should the sales team contact first?"

**Solution Approach:**
Build a supervised binary classification model trained on historical lead data, where each lead's outcome (converted or not converted) is known. The trained model will output a probability score for new leads, which is then mapped to a 0–100 lead score and a priority bucket (Hot / Warm / Cold).

---

## 3. Objectives & Goals

**Primary Objectives:**
1. Build a binary classification model that predicts lead conversion with at least 80% accuracy.
2. Generate a lead score (0–100) for each lead representing conversion probability.
3. Identify the top 5–10 features that most strongly influence conversion.
4. Compare at least 2 ML algorithms and justify the final model choice.

**Secondary Goals:**
- Document the end-to-end approach in a clean, reproducible notebook.
- Visualize key insights (feature importance, conversion patterns, confusion matrix).
- Produce a deliverable that could realistically be handed to a sales team.

---

## 4. Scope

**In Scope:**
- Loading and exploring the Kaggle Lead Scoring dataset.
- Data cleaning (missing values, duplicates, inconsistent categories, "Select" placeholders).
- Exploratory Data Analysis with visualizations.
- Feature engineering (encoding, scaling, feature selection).
- Training and tuning multiple classification models.
- Model evaluation with accuracy, precision, recall, F1, and ROC-AUC.
- Lead score generation and priority bucketing (Hot / Warm / Cold).
- Final notebook with markdown explanations and conclusions.

**Out of Scope:**
- Deploying the model as a web application (that is the Advanced Level task).
- Building a UI / Streamlit dashboard.
- Real-time integration with a CRM.
- Deep learning / neural network approaches.
- Collecting or scraping new data.

---

## 5. Dataset Specification

| Attribute | Details |
|---|---|
| Dataset Name | Lead Scoring Dataset |
| Source | Kaggle |
| Dataset URL | https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset |
| Uploaded By | Amrita Chatterjee |
| File Format | CSV (Leads.csv) |
| Total Records | ~9,240 leads |
| Total Features | 37 columns (mix of categorical and numerical) |
| Target Variable | "Converted" — Binary (1 = Converted, 0 = Not Converted) |
| Domain | Education / EdTech — X Education company lead data |
| Backup Dataset | https://www.kaggle.com/datasets/ashydv/leads-dataset |

**How to download:** Sign in to Kaggle → open the dataset URL above → click "Download" → unzip to get `Leads.csv`. In Google Colab, you can also use the Kaggle API with `kaggle datasets download -d amritachatterjee09/lead-scoring-dataset`.

### Key Features to Use

| Feature | Type | Why It Matters |
|---|---|---|
| Lead Source | Categorical | Channel through which lead arrived (Google, Direct Traffic, Referral) — strong conversion signal |
| Total Time Spent on Website | Numerical | Engagement metric — higher time correlates with higher conversion intent |
| TotalVisits | Numerical | Number of times the lead visited the website |
| Page Views Per Visit | Numerical | Depth of engagement per session |
| Last Activity | Categorical | Most recent lead activity (email opened, SMS sent, form submitted) |
| Lead Origin | Categorical | API, Landing Page Submission, Lead Add Form, etc. |
| Specialization | Categorical | Course/specialization the lead is interested in |
| Occupation | Categorical | Working professional, unemployed, student — strongly predictive |
| Tags | Categorical | Sales-team-applied label — high signal but use with caution (may leak target) |
| Converted (Target) | Binary | 1 if the lead was converted to a paying customer, 0 otherwise |

**⚠️ Data leakage warning:** Columns like `Tags`, `Last Notable Activity`, and `Lead Quality` are populated by the sales team AFTER engaging with the lead and may leak the target. Treat them carefully — either drop them or document the assumption that they are available at prediction time.

---

## 6. Functional Requirements

### 6.1 Data Preprocessing
- Load dataset using Pandas `read_csv`.
- Identify and remove columns with > 40% missing values.
- Replace "Select" placeholder values (a known artifact of the form UI) with NaN.
- Impute missing values: mode for categorical, median for numerical.
- Drop irrelevant identifier columns (Prospect ID, Lead Number).
- Remove duplicate rows if any.
- Handle outliers in numerical columns using IQR capping.

### 6.2 Exploratory Data Analysis (EDA)
- Compute and display the overall conversion rate (baseline).
- Plot distribution of the target variable (Converted).
- Visualize conversion rate by Lead Source, Lead Origin, Last Activity, and Occupation.
- Plot histograms / box plots for numerical features.
- Generate a correlation heatmap for numerical features.
- Document at least 3 key insights from EDA in markdown cells.

### 6.3 Feature Engineering
- Apply One-Hot Encoding to nominal categorical features.
- Apply Label Encoding to binary categorical features (Yes/No).
- Scale numerical features using StandardScaler or MinMaxScaler.
- Perform feature selection using Recursive Feature Elimination (RFE) or feature importance from a tree-based model.
- Split data into training (70%) and testing (30%) sets with stratification on the target.

### 6.4 Model Training

| Model | Why Use It | Role |
|---|---|---|
| Logistic Regression | Simple, interpretable, gives direct probability scores (perfect for 0–100 scoring) | Baseline + Primary scoring model |
| Decision Tree | Easy to visualize, captures non-linear patterns | Comparison model |
| Random Forest | Robust to overfitting, strong feature importance | Comparison model |
| XGBoost (Optional) | Gradient boosting — typically highest accuracy on tabular data | Bonus / stretch goal |

### 6.5 Model Evaluation

| Metric | What It Measures | Target |
|---|---|---|
| Accuracy | Overall correctness of predictions | ≥ 80% |
| Precision | Of leads predicted hot, how many actually converted | ≥ 75% |
| Recall | Of all converted leads, how many we identified | ≥ 75% |
| F1-Score | Harmonic mean of Precision and Recall | ≥ 0.75 |
| ROC-AUC | Model's discrimination ability across thresholds | ≥ 0.85 |
| Confusion Matrix | Visual breakdown of TP, FP, TN, FN | Required deliverable |

### 6.6 Lead Scoring Output
- Use `predict_proba` to get conversion probability for each lead.
- Multiply by 100 to produce a Lead Score (0–100).
- Bucket leads into priority tiers:
  - **Hot Lead** → Score ≥ 70 (call immediately)
  - **Warm Lead** → Score 40–69 (nurture with email)
  - **Cold Lead** → Score < 40 (de-prioritize)
- Output a final DataFrame with: Lead ID, predicted score, bucket, top 3 contributing features.

---

## 7. Technical Requirements

**Tools & Platform:**
- Python 3.9+
- Google Colab (preferred) or Jupyter Notebook
- GitHub for version control and submission link

**Python Libraries:**
- pandas — data manipulation
- numpy — numerical operations
- matplotlib & seaborn — visualization
- scikit-learn — preprocessing, models, metrics
- xgboost (optional) — gradient boosting model

---

## 8. Methodology / Approach

The project follows the CRISP-DM workflow:
1. **Business Understanding** — Define what "a good lead" means.
2. **Data Understanding** — Inspect shape, dtypes, missing values, target distribution.
3. **Data Preparation** — Clean, impute, encode, scale.
4. **Modeling** — Train baseline (Logistic Regression) → compare with Random Forest and Decision Tree → optionally XGBoost.
5. **Evaluation** — Compare on test set using all 5 metrics. Pick best based on F1 and interpretability.
6. **Deployment Prep** — Generate lead scores and document for future Streamlit/Flask extension.

---

## 9. Timeline & Milestones

Total effort: ~6 working days (15–20 focused hours).

| Day | Milestone | Deliverable |
|---|---|---|
| 1 | Setup & Data Loading | Kaggle dataset downloaded, Colab notebook created, data loaded |
| 2 | EDA & Preprocessing | Missing values handled, "Select" placeholders cleaned, charts plotted |
| 3 | Feature Engineering | Encoding, scaling, train/test split, leakage columns dropped |
| 4 | Model Training | Logistic Regression + Random Forest + Decision Tree trained |
| 5 | Evaluation & Scoring | Metrics computed, confusion matrix, lead scores generated |
| 6 | Documentation & Submission | Notebook polished, README written, GitHub link submitted |

---

## 10. Deliverables

- Jupyter / Colab notebook (`.ipynb`) with the full pipeline and markdown explanations.
- `README.md` describing the project, dataset, how to run, key findings.
- Final lead score CSV with: Lead ID, Lead Score, Priority Bucket.
- Visualizations: EDA plots, feature importance chart, ROC curve, confusion matrix.
- Short written summary explaining chosen model, top features, business recommendations.
- GitHub repository link or Colab share link submitted via the TechnoHacks portal.

---

## 11. Success Criteria

- Notebook runs end-to-end without errors on a fresh Colab kernel.
- Final model achieves ≥ 80% accuracy and ≥ 0.75 F1-score on the test set.
- At least 2 ML models compared with clear justification for the final choice.
- Top features influencing conversion are identified and visualized.
- Each lead has an assigned 0–100 score and a priority bucket.
- All code includes comments and the notebook tells a clear story via markdown.
- Submission accepted via the TechnoHacks portal and approved by the mentor.

---

## 12. Risks & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Class imbalance (~38% conversions) | Model biased toward majority class | Stratified split, `class_weight='balanced'`, evaluate with F1 |
| Data leakage from sales-applied columns | Unrealistically high accuracy | Drop "Tags" / "Lead Quality" or document assumption |
| High cardinality categoricals (City) | Exploding one-hot dimensions | Group rare categories into "Other" |
| Overfitting | Poor generalization | Cross-validation + L2 regularization |
| "Select" placeholder as valid category | Misleading model signal | Replace "Select" with NaN explicitly |

---

## 13. References

- **Primary Dataset:** https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset
- **Reference Notebook:** https://www.kaggle.com/code/ashydv/lead-scoring-logistic-regression
- **End-to-End Example:** https://www.kaggle.com/code/celocruz/lead-scoring-end-to-end-project
- **Scikit-learn Documentation:** https://scikit-learn.org/stable/
- **Program Brief:** TechnoHacks Internship Program — Data Science Project Task Guide (Medium Level)

---

*— End of Document —*

---
