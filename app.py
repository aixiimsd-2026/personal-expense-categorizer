from flask import Flask, render_template, request
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Create Flask application
app = Flask(__name__)


# Load the expense dataset
data = pd.read_csv("expenses.csv")

X = data["Description"]
y = data["Category"]


# Create Machine Learning model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Train the model
model.fit(X, y)


# Store expense history
expense_history = []


# Home page
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    expense = ""
    amount = ""

    if request.method == "POST":

        expense = request.form["expense"]
        amount = request.form["amount"]

        if expense.strip() != "":

            # Predict category
            prediction = model.predict([expense])[0]

            # Calculate confidence
            probabilities = model.predict_proba([expense])[0]
            confidence = round(max(probabilities) * 100, 2)

            # Add expense to history
            expense_history.append({
                "expense": expense,
                "amount": amount,
                "category": prediction,
                "confidence": confidence
            })

    # Calculate total spending for each category
    category_totals = {}

    for item in expense_history:

        category = item["category"]
        amount_value = float(item["amount"])

        if category not in category_totals:
            category_totals[category] = 0

        category_totals[category] += amount_value

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        expense=expense,
        amount=amount,
        expense_history=expense_history,
        category_totals=category_totals
    )


# Start the website
if __name__ == "__main__":
    app.run(debug=True)
