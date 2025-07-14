# This is a simple Flask web application for Titanic survival prediction.
# It uses Decision Tree and Random Forest classifiers to predict survival based on passenger data.
# It includes a web interface for user input and displays the prediction results.
#
# Project structure:
# data_science/
# │
# ├── app.py                   ← Flask web app using your ML model
# ├── requirements.txt         ← Python dependencies
# ├── templates/
# │   └── index.html           ← Web form + display results
# ├── titanic_prediction.py    ← (Optional) Keep this for modular use


from flask import Flask, request, render_template
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# Train models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt_model = DecisionTreeClassifier().fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Pclass = int(request.form['Pclass'])
        Sex = 1 if request.form['Sex'] == 'female' else 0
        Age = float(request.form['Age'])
        Fare = float(request.form['Fare'])

        input_data = [[Pclass, Sex, Age, Fare]]

        dt_pred = dt_model.predict(input_data)[0]
        rf_pred = rf_model.predict(input_data)[0]

        return render_template(
            'index.html',
            dt_result="Survived" if dt_pred else "Did not survive",
            rf_result="Survived" if rf_pred else "Did not survive"
        )
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000)) # Default to port 5000 if not set by Render
    app.run(host='0.0.0.0', port=port, debug=True)  # Set debug=True for development
