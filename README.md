# 🏦 Loan Approval Prediction System

A Machine Learning-based web application that predicts whether a loan application will be **approved or rejected** based on financial and personal details.

---

## 🚀 Features

- 🔍 Predicts loan approval using Logistic Regression
- 📊 Uses real-world loan approval dataset
- ⚙️ Handles both categorical and numerical features
- 📈 Displays prediction with confidence score
- 🌐 Interactive UI built with Streamlit

---

## 🧠 Tech Stack

- Python 🐍
- Scikit-learn
- Pandas
- NumPy
- Streamlit

---

## 📊 Input Features

- Number of Dependents
- Education
- Self Employment Status
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Asset Value

---

## 🎯 Output

- ✅ Loan Approved  
- ❌ Loan Rejected  
- 📈 Confidence Score (%)

---

## ⚙️ How It Works

1. Data preprocessing (cleaning, encoding, scaling)
2. Logistic Regression model training
3. User inputs data via Streamlit UI
4. Model predicts loan approval with probability

---

## ▶️ Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
Loan_Prediction/
│
├── app.py
├── train_model.py
├── loan_approval_dataset.csv
├── requirements.txt
├── README.md
└── .gitignore
```
