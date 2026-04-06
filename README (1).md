# 📡 Telco Customer Churn Prediction System

<div align="center">

> **Predicting customer churn before it happens — because retaining a customer is always cheaper than losing one.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-11557C?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

<br/>

| 📊 Dataset | 🎯 Task | 🏆 Best Model | 📈 Focus Metric |
|:---:|:---:|:---:|:---:|
| 7,043 customers · 50 features | Binary Classification | Logistic Regression | Recall (Churn Class) |

</div>

---

## 💡 Why I Built This

I was going through customer behavior datasets online and one thing kept hitting me — telecom companies lose **millions every year** just because they can't figure out *who is about to leave* before it's too late.

That got me thinking: can a machine learn to predict churn the way an experienced customer care rep "just knows" when a customer is frustrated?

So I built this. End-to-end. Learning as I went.

This isn't just a model — it's a complete ML pipeline that handles real-world messiness: **data leakage, missing values, encoding pitfalls, generalization testing** — the stuff that actually matters when you take ML beyond tutorials.

---

## 🎯 Problem Statement

In the telecom industry, **customer churn** (a customer discontinuing service) is one of the biggest revenue drains. Acquiring a new customer costs **5–7× more** than retaining an existing one.

**Real numbers from this dataset:**
- 📉 **26.5% overall churn rate** — more than 1 in 4 customers leaves
- ⏳ Churned customers had an **average tenure of 18 months** vs. 37.6 months for loyal customers
- 💸 Churned customers paid **₹74.44/month on average** vs. ₹61.27 for those who stayed

The goal:
- ✅ **Predict whether a customer will churn** (Yes / No)
- ✅ **Identify the key drivers** behind churn
- ✅ **Build a pipeline that generalizes** to completely unseen data

---

## 📂 Dataset Description

> **Source:** IBM Telco Customer Churn Dataset — California Q3 sample  
> **Size:** 7,043 customers · 50 columns  
> **Split:** 4,225 training · 1,409 testing (stratified 75/25)

| Category | Columns | Examples |
|---|---|---|
| 👤 Demographics | 8 | Age, Gender, Married, Dependents |
| 🌐 Services | 13 | Internet Type, Streaming TV, Online Security |
| 💳 Billing | 7 | Monthly Charge, Payment Method, Paperless Billing |
| 📋 Account | 4 | Tenure in Months, Contract, Offer |
| 🏷️ Post-Churn (DROPPED) | 7 | Churn Score, Churn Reason, CLTV, Total Revenue |
| 🎯 Target | 1 | **Churn** (Yes = 26.5%) |

### ⚠️ Class Distribution

```
Not Churned (No)  ████████████████████████████░░░░  73.5%  →  5,174 customers
Churned    (Yes)  ████████░░░░░░░░░░░░░░░░░░░░░░░░  26.5%  →  1,869 customers
```

---

## 🛠️ Tech Stack

```python
Python 3.10+
├── pandas          # Data manipulation
├── numpy           # Numerical operations
├── scikit-learn    # ML models, preprocessing, evaluation
│   ├── LogisticRegression
│   ├── RandomForestClassifier
│   ├── StandardScaler
│   └── train_test_split (stratified)
├── matplotlib      # EDA visualizations
└── joblib          # Model serialization
```

---

## 🔄 Complete Project Workflow

```
📁 Raw Data (telco.csv — 7,043 rows, 50 cols)
           │
           ▼
🚨 Data Leakage Removal  ──── Drop: Churn Score, Churn Reason,
           │                        Churn Category, Customer Status,
           │                        CLTV, Total Revenue, Customer ID
           ▼
🧹 Missing Value Handling  ── Numerical  → Median imputation
           │                  Categorical → "Unknown" fill
           ▼
⚙️  Feature Engineering  ───── Avg_Charge_Per_Tenure = Total Charges / (Tenure + 1)
           │
           ▼
🔢 One-Hot Encoding  ────────── pd.get_dummies(drop_first=True)
           │                    + Column alignment (train ↔ test)
           ▼
✂️  Train-Test Split  ───────── 80% Train / 20% Test (stratify=y)
           │
           ▼
📏 Feature Scaling  ─────────── StandardScaler
           │                    fit_transform(train) | transform(test)
           ▼
🤖 Model Training
           ├── Logistic Regression (max_iter=4000)
           └── Random Forest Classifier
           │
           ▼
📊 Evaluation  ──────────────── Accuracy, Precision, Recall, F1
           │                    Confusion Matrix
           ▼
🌍 Generalization Test  ─────── Same pipeline → test.csv (unseen)
           │
           ▼
🔍 Feature Importance  ──────── Top predictors via Random Forest
           │
           ▼
💾 Model Saved  ─────────────── churn_model.pkl
```

---

## 🚨 Handling Data Leakage — The Most Important Part

This was the biggest learning of this entire project.

When I first trained the model *without* removing certain columns, I was getting accuracy above **95%**. I was excited — I thought I had built something incredible.

Then I stopped and thought: *"Wait — how would a model know the Churn Reason if the customer hasn't churned yet?"*

That's when it hit me. Columns like `Churn Score`, `Churn Reason`, `Churn Category`, `Customer Status`, `CLTV`, and `Total Revenue` are all **generated after** the churn event. They're future information. In real deployment, you'd never have this data *before* deciding to retain a customer.

The model wasn't smart — it was cheating. Reading the answer from the question paper.

```python
cols_to_drop = [
    "Customer ID",       # Unique identifier — no predictive value
    "Churn Category",    # ⛔ POST-CHURN: filled only after customer leaves
    "Churn Reason",      # ⛔ POST-CHURN: filled only after customer leaves
    "Customer Status",   # ⛔ POST-CHURN: literally says "Churned" or "Stayed"
    "Churn Score",       # ⛔ LEAKAGE: derived from churn outcome
    "CLTV",              # ⛔ LEAKAGE: calculated using future revenue data
    "Total Revenue"      # ⛔ LEAKAGE: includes revenue from the churn period
]
```

**After removing them:** Accuracy dropped to a realistic range. But now the model is *honest*. It performs well on data it has never seen — which is the only thing that actually matters in production.

> 📌 **Lesson:** A realistic model that generalizes is always more valuable than a "perfect" model that cheats.

---

## 🧹 Data Preprocessing

### Missing Value Strategy

```python
# Numerical columns → Median (robust to outliers)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns → "Unknown" (preserves category integrity)
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")
```

**Why median over mean for numerical?**  
Outliers in billing data (e.g., extreme `Total Charges`) skew the mean significantly. Median stays stable.

---

## ⚙️ Feature Engineering

```python
df["Avg_Charge_Per_Tenure"] = df["Total Charges"] / (df["Tenure in Months"] + 1)
```

**Why this feature matters:**

A customer paying ₹500/month for 1 month is completely different from one paying ₹500/month for 24 months. Raw `Monthly Charge` alone doesn't capture this contrast. `Avg_Charge_Per_Tenure` gives the model a sense of *spending relative to loyalty* — and it ended up being one of the **top 5 most important features**.

The `+1` prevents division-by-zero for brand-new customers (Tenure = 0).

---

## 🔢 Encoding & Column Alignment

```python
# One-Hot Encoding — converts categorical to numerical
X_train = pd.get_dummies(X_train, drop_first=True)
X_test  = pd.get_dummies(X_test,  drop_first=True)

# Alignment — prevents crashes from column mismatch
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# For the second test file — reindex to match training columns exactly
X_test1 = X_test1.reindex(columns=train_columns, fill_value=0)
```

**Why `drop_first=True`?**  
Avoids the **dummy variable trap** — when two binary columns perfectly predict each other, introducing multicollinearity that breaks linear models.

**Why `.align()` and `.reindex()`?**  
One-hot encoding of train and test separately can produce different column counts (e.g., a rare city only appears in training). Without alignment, the model crashes silently or produces garbage predictions.

---

## 📊 Train-Test Split & Scaling

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,       # Maintains 73.5% / 26.5% ratio in both splits
    random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Learn mean/std FROM training data
X_test  = scaler.transform(X_test)        # Apply same scale — NO fitting on test
```

**Critical:** `scaler.transform()` on test data — never `fit_transform()`. Fitting the scaler on test data is a subtle form of data leakage where test statistics contaminate preprocessing.

---

## 🤖 Models

### 1️⃣ Logistic Regression

```python
model = LogisticRegression(max_iter=4000)
model.fit(X_train, y_train)
```

Simple, interpretable, and fast. `max_iter=4000` ensures convergence on high-dimensional encoded data. **Winner for this use case** — higher recall on the churn class.

### 2️⃣ Random Forest Classifier

```python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
```

Ensemble method — trains multiple decision trees and aggregates their votes. Slightly higher accuracy, but used primarily here to extract **feature importances**.

---

## 📈 Evaluation Metrics

For churn prediction, **Recall on the churn class is the most critical metric.**

```
False Negative = Predicted "Won't Churn" but customer actually LEFT
→ The business lost a customer they could have retained
→ This is the most expensive mistake
```

So we care more about catching every potential churner (high recall) than being perfectly precise.

| Metric | What It Measures |
|---|---|
| **Accuracy** | Overall correctness across both classes |
| **Precision** | Of all predicted churners — how many actually churned |
| **Recall** ⭐ | Of all actual churners — how many did we catch |
| **F1-Score** | Harmonic mean of Precision and Recall |

---

## 🏆 Results

### On Internal Test Set (20% split from training data)

| Model | Accuracy | Recall (Churn) | F1 (Churn) | Decision |
|---|---|---|---|---|
| **Logistic Regression** | ~82% | ✅ **Higher** | Balanced | ✅ **Chosen** |
| Random Forest | ~84% | Slightly lower | Higher precision | Used for feature importance |

### On Completely Unseen Data (`test.csv` — 1,409 records)

Both models were validated on `test.csv` — a file the pipeline had **never seen during training**. The same preprocessing steps were applied:
- Same column drops
- Same imputation logic
- Same one-hot encoding → `.reindex()` to match training columns
- Same scaler (`.transform()` only)

**Result: The model generalized well.** Performance on unseen data closely matched internal test performance — confirming this is a real pipeline, not an overfit experiment.

---

## 🌟 Feature Importance

*Extracted from Random Forest — top predictors of churn:*

| Rank | Feature | Type | Insight |
|---|---|---|---|
| 🥇 1 | Satisfaction Score | Original | Dissatisfied customers leave — obvious in hindsight |
| 🥈 2 | Tenure in Months | Original | Loyalty is self-reinforcing — long-term customers rarely leave |
| 🥉 3 | Total Charges | Original | Higher lifetime spend = more invested in staying |
| 4 | Monthly Charge | Original | High monthly bills actively push customers to competitors |
| 5 | **Avg_Charge_Per_Tenure** | ✨ **Engineered** | Spend-to-loyalty ratio — captures early high-paying customers at risk |

```python
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=train_columns
).sort_values(ascending=False)
```

> The engineered feature `Avg_Charge_Per_Tenure` breaking into the Top 5 validated that feature engineering was worth the effort.

---

## 🔍 Key EDA Insights

Real patterns discovered from the data — not assumptions:

### 📋 Contract Type → Biggest Churn Driver

| Contract | Churn Rate |
|---|---|
| Month-to-Month | **45.8%** 🔴 |
| One Year | 10.7% 🟡 |
| Two Year | **2.5%** 🟢 |

> Customers on month-to-month contracts churn at **18× the rate** of two-year contract customers. No commitment = easy exit.

### ⏳ Tenure → Early Months Are Critical

- Churned customers: **avg 18 months** tenure
- Loyal customers: **avg 37.6 months** tenure

The first 12 months of a customer's lifecycle are the most vulnerable. Onboarding and early engagement matter enormously.

### 💸 Monthly Charge → Higher Bills = Higher Risk

- Churned customers paid **₹74.44/month** on average
- Loyal customers paid **₹61.27/month** on average

A **21% higher monthly charge** for churned customers. Price sensitivity is real — especially when competitors are offering better deals (the #1 churn reason in this dataset).

### 📡 Top Churn Reasons

| Reason | Count |
|---|---|
| Competitor had better devices | 313 |
| Competitor made better offer | 311 |
| Attitude of support person | 220 |
| Don't know | 130 |
| Competitor offered more data | 117 |

> **Competition is the #1 enemy.** Followed closely by support quality — customers leave bad experiences as much as they leave bad prices.

---

## ⚠️ Challenges & How I Debugged Them

### Challenge 1 — The 95% Accuracy Trap 🕳️
**What happened:** Early model was hitting 95%+ accuracy. Seemed amazing.  
**What was wrong:** `Churn Score`, `Customer Status`, and `Churn Reason` columns were included — all assigned *after* a customer churns.  
**Fix:** Identified leakage via feature importance (these columns dominated). Dropped them all.  
**Result:** More realistic accuracy, but a model that actually works in the real world.

### Challenge 2 — Column Mismatch After Encoding 💥
**What happened:** `X_train` had 87 columns after `get_dummies`. `X_test` had 83. Model crashed.  
**What was wrong:** Rare categories (e.g., an offer type or city only in training set) created extra columns in one dataset.  
**Fix:** Used `.align(join='left')` to force test to match train's column structure.

### Challenge 3 — Applying Pipeline to New Data 🔄
**What happened:** When running on `test.csv` (a completely new file), preprocessing had to be replicated exactly.  
**Tricky parts:**
- Had to reindex columns using `train_columns` (saved before scaling)
- Had to use `.transform()` on the same fitted scaler — never re-fit
- Had to apply the same `cols_to_drop` and same imputation logic

**Fix:** Saved `train_columns` and `scaler` before training. Replicated preprocessing step by step.

### Challenge 4 — Class Imbalance Awareness ⚖️
**What happened:** 73.5% of data is "Not Churned" — a naive model could achieve 73.5% accuracy by just predicting "No" every time.  
**Fix:** Used `stratify=y` in the split. Focused on **per-class recall** in the classification report instead of raw accuracy.

---

## 🚀 Future Improvements

- [ ] **XGBoost / LightGBM** — likely to outperform both current models
- [ ] **Hyperparameter tuning** — GridSearchCV or Optuna
- [ ] **SMOTE** — synthetic minority oversampling for class imbalance
- [ ] **SHAP values** — model explainability beyond feature importances
- [ ] **FastAPI endpoint** — serve `churn_model.pkl` as a REST API
- [ ] **Streamlit dashboard** — interactive UI for business users
- [ ] **MLflow** — experiment tracking and model versioning
- [ ] **Docker container** — for reproducible deployment

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/pedadarohan-dot/Telco-Churn-Prediction.git
cd Telco-Churn-Prediction
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add data files

Place these in the project root:
```
Telco-Churn-Prediction/
├── train.csv
├── test.csv
└── telco.csv
```

### 4. Run the full pipeline

```bash
python data_handling.py
```

This will:
- Clean and preprocess the data
- Train Logistic Regression + Random Forest
- Print classification reports for both models
- Show confusion matrix and EDA plots
- Test on unseen `test.csv`
- Save the model as `churn_model.pkl`

### 5. Use the saved model

```python
import joblib
import pandas as pd

# Load model
model = joblib.load("churn_model.pkl")

# Apply same preprocessing pipeline to new data, then:
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

---

## 📁 Project Structure

```
Telco-Churn-Prediction/
│
├── 📄 data_handling.py      ← Full ML pipeline (cleaning → training → evaluation)
├── 📊 train.csv             ← Training dataset (4,225 records)
├── 📊 test.csv              ← Generalization test dataset (1,409 records)
├── 📊 telco.csv             ← Original full dataset (7,043 records, 50 columns)
├── 🤖 churn_model.pkl       ← Saved Logistic Regression model
│
├── 📋 requirements.txt      ← Project dependencies
├── 📋 CHANGELOG.md          ← Version history
├── 🤝 CONTRIBUTING.md       ← How to contribute
├── ⚖️  LICENSE               ← MIT License
└── 📖 README.md             ← You are here
```

---

## 🧠 What Makes This Project Stand Out

| Aspect | What Was Done |
|---|---|
| 🚨 **Data Leakage** | Identified and removed 6 leakage-prone columns — not just mentioned, actually debugged |
| ⚙️ **Feature Engineering** | Created `Avg_Charge_Per_Tenure` — ranked in Top 5 most important features |
| 🔢 **Encoding** | Handled dummy variable trap + column mismatch between train/test |
| 🌍 **Generalization** | Validated on a fully separate `test.csv` with identical preprocessing |
| 📊 **Right Metrics** | Focused on Recall for churn class — not just accuracy |
| 💾 **Deployable** | Model saved with `joblib` — ready to serve via API |

---

## 📬 Connect

Built with 💻 + ☕ by **Rohan**

[![LinkedIn](https://www.linkedin.com/in/rohan-pedada-b6b250380/)
[![GitHub](https://github.com/pedadarohan-dot)

---

<div align="center">

*If this project helped you or inspired you — drop a ⭐. It means a lot.*

**Onwards and upwards 🚀**

</div>
