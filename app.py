
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model pipeline
model = pickle.load(open('pipe.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and sanitize user inputs
        pclass = int(request.form['pclass'])
        sex = request.form['sex'].strip().lower()
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked'].strip().upper()

        # Create DataFrame with correct column names and order
        input_df = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked': [embarked]
        })

        # Predict using the pipeline
        model_pred = model.predict(input_df)
        prediction = model_pred[0]
        result = "Survived" if prediction == 1 else "Not Survived"
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run()