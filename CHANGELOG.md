# 📋 Changelog

All notable changes to the **Telco Customer Churn Prediction System** are documented here.

This project follows [Semantic Versioning](https://semver.org/) — `MAJOR.MINOR.PATCH`.

---

## [1.0.0] — 2025 — Initial Release 🎉

### Added
- Full end-to-end ML pipeline in `data_handling.py`
- Data leakage detection and removal (dropped `Churn Score`, `Churn Category`, `Churn Reason`, `Customer Status`, `CLTV`, `Total Revenue`)
- Missing value handling — median imputation for numerical, "Unknown" fill for categorical
- Feature engineering — `Avg_Charge_Per_Tenure = Total Charges / (Tenure in Months + 1)`
- One-Hot Encoding with `drop_first=True` to avoid dummy variable trap
- Train/test column alignment using `.align()` and `.reindex()`
- Stratified 80/20 train-test split
- StandardScaler feature scaling (fit on train, transform on test)
- Logistic Regression model (max_iter=4000)
- Random Forest Classifier model
- Full evaluation with classification report and confusion matrix
- Generalization test on completely unseen `test.csv`
- Feature importance extraction from Random Forest
- EDA plots — churn distribution, contract analysis, tenure analysis, monthly charge analysis
- Saved trained model as `churn_model.pkl` via joblib

### Project Files
- `data_handling.py` — main pipeline
- `train.csv` — training data (4,225 records)
- `test.csv` — unseen test data (1,409 records)
- `telco.csv` — original full dataset (7,043 records, 50 columns)
- `churn_model.pkl` — saved Logistic Regression model
- `requirements.txt` — project dependencies
- `LICENSE` — MIT License
- `CONTRIBUTING.md` — contribution guidelines
- `README.md` — full project documentation

---

## [Upcoming] — v1.1.0

### Planned
- [ ] Add XGBoost and LightGBM models
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] SMOTE for class imbalance
- [ ] SHAP value explainability
- [ ] FastAPI REST endpoint for model serving
- [ ] Streamlit web UI for business users

---

*Built by Rohan — Onwards and upwards 🚀*
