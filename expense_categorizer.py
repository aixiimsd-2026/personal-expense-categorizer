import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Load dataset
data = pd.read_csv("expenses.csv")

# Input and output
X = data["Description"]
y = data["Category"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create Machine Learning model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Train the model
model.fit(X_train, y_train)


# Predict test data
y_pred = model.predict(X_test)


# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# Display results
print("\n========== MODEL EVALUATION ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Macro Precision:", round(precision * 100, 2), "%")
print("Macro Recall:", round(recall * 100, 2), "%")
print("Macro F1-score:", round(f1 * 100, 2), "%")


# Classification report
print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# Confusion matrix
print("\n========== CONFUSION MATRIX ==========")

categories = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=categories
)

print("Categories:", categories)
print(cm)