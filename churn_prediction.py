
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Original dataset shape:", data.shape)

# ==========================================
# STEP 2: DATA CLEANING
# ==========================================

# Remove customer ID
data.drop(columns=["customerID"], inplace=True)

# Convert TotalCharges into numeric values
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)

# Fill missing TotalCharges with 0
data["TotalCharges"] = data["TotalCharges"].fillna(0)

# Remove duplicate rows
data.drop_duplicates(inplace=True)

print("\nCleaned dataset shape:", data.shape)

print("\nChurn counts:")
print(data["Churn"].value_counts())

print("\nRemaining missing values:", data.isnull().sum().sum())

# ==========================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# Chart 1: Customer Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (No = Stayed, Yes = Left)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Chart 2: Churn by Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Chart 3: Monthly Charges by Churn
plt.figure(figsize=(7, 4))
sns.boxplot(data=data, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn (No = Stayed, Yes = Left)")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()

print("\nExploratory data analysis completed!")

# ==========================================
# STEP 4: PREPARE DATA FOR MACHINE LEARNING
# ==========================================

# Input features
X = data.drop(columns=["Churn"])

# Target variable: No = 0, Yes = 1
y = data["Churn"].map({"No": 0, "Yes": 1})

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(
    include=["number"]
).columns

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns

# Preprocess numeric and categorical data
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_columns),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_columns
    )
])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# ==========================================
# STEP 5: CREATE MACHINE LEARNING MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}

best_model = None
best_accuracy = -1
best_model_name = ""

# Store results for comparison
results = []

# ==========================================
# STEP 6: TRAIN AND EVALUATE MODELS
# ==========================================

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Make predictions
    predictions = pipeline.predict(X_test)

    # Churn probabilities
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "ROC-AUC": round(roc_auc, 3)
    })

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("ROC-AUC:", round(roc_auc, 3))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Stayed", "Churned"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Keep the model with the highest test accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_model_name = name

# ==========================================
# STEP 7: COMPARE MODELS
# ==========================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 45)
print("MODEL COMPARISON")
print("=" * 45)

print(results_df.to_string(index=False))

# ==========================================
# STEP 8: SAVE BEST MODEL
# ==========================================

joblib.dump(best_model, "churn_model.pkl")

print("\nBest model:", best_model_name)
print("Best accuracy:", round(best_accuracy * 100, 2), "%")
print("Model saved successfully as churn_model.pkl")

print("\nCustomer Churn Prediction project completed!")



