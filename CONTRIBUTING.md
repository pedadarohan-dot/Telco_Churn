# 🤝 Contributing to Telco Customer Churn Prediction System

First off — thank you for taking the time to contribute! 🎉

This project is open to improvements, bug fixes, new ideas, and better approaches. Whether you're a beginner or a senior ML engineer, your contribution is valued.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

---

## 🧭 Code of Conduct

Be respectful. Be constructive. Be kind.

This is a learning-focused project. Everyone is welcome — regardless of experience level.

---

## 💡 How Can I Contribute?

### 🐛 Reporting Bugs
- Open a GitHub Issue
- Include what you expected vs. what happened
- Include your Python version and OS

### 🌟 Suggesting Enhancements
- Open a GitHub Issue with the label `enhancement`
- Describe the idea clearly and why it adds value

### 🔧 Code Contributions
Good first issues to work on:
- [ ] Add XGBoost / LightGBM model comparison
- [ ] Add SHAP value visualizations
- [ ] Build a Streamlit prediction UI
- [ ] Add hyperparameter tuning with GridSearchCV
- [ ] Add SMOTE for class imbalance handling
- [ ] Write unit tests for preprocessing functions
- [ ] Create a FastAPI endpoint to serve the saved model

---

## 🚀 Getting Started

```bash
# 1. Fork the repo on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/Telco-Churn-Prediction.git
cd Telco-Churn-Prediction

# 3. Create a new branch (NEVER work on main directly)
git checkout -b feature/your-feature-name

# 4. Set up the environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 5. Make your changes and commit
git add .
git commit -m "feat: add SHAP value visualization"

# 6. Push to your fork
git push origin feature/your-feature-name

# 7. Open a Pull Request on GitHub 🎉
```

---

## ✅ Pull Request Process

1. Make sure your code **runs without errors**
2. Follow the existing **code style** (see below)
3. Update the **README** if your change affects usage
4. Keep PRs **focused** — one feature/fix per PR
5. Add a clear **description** of what you changed and why

---

## 🎨 Style Guidelines

- Use **PEP 8** formatting
- Add **comments** to non-obvious logic
- Use **descriptive variable names** — no `x1`, `tmp2`, etc.
- Keep functions small and single-purpose
- Use f-strings over `.format()` or `%`

```python
# ✅ Good
churn_rate = (df["Churn Label"] == "Yes").mean()
print(f"Overall churn rate: {churn_rate:.1%}")

# ❌ Avoid
x = (df["Churn Label"] == "Yes").mean()
print("Rate: " + str(x))
```

---

## 🙏 Thank You

Every contribution — big or small — makes this project better.

If you found this project useful, please consider giving it a ⭐ on GitHub. It means a lot!

Onwards and upwards 🚀
