import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("loan_approval_dataset.csv")

# Fix column names
df.columns = df.columns.str.strip().str.lower()

# Drop ID column
if "loan_id" in df.columns:
    df.drop("loan_id", axis=1, inplace=True)

# Handle missing values
df.ffill(inplace=True)

# Clean categorical
df["education"] = df["education"].astype(str).str.strip().str.lower()
df["self_employed"] = df["self_employed"].astype(str).str.strip().str.lower()

# Encode categorical
le_edu = LabelEncoder()
le_emp = LabelEncoder()

df["education"] = le_edu.fit_transform(df["education"])
df["self_employed"] = le_emp.fit_transform(df["self_employed"])

# Clean target
df["loan_status"] = df["loan_status"].astype(str).str.strip().str.lower()
df["loan_status"] = df["loan_status"].map({
    "approved": 1,
    "rejected": 0
})

# Remove invalid rows
df = df.dropna(subset=["loan_status"])

# Features & target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# 🔥 Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 Scale AFTER split
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save files
pickle.dump(model, open("loan_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le_edu, open("le_edu.pkl", "wb"))
pickle.dump(le_emp, open("le_emp.pkl", "wb"))

# Debug
print("✅ Model trained successfully!")
print("\nTarget distribution:")
print(df["loan_status"].value_counts())