import csv
import numpy as np
from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_list = []  # Create an empty list to store inputs

        # Retrieve input values from the form and append them to the list
        for key in request.form.keys():
            input_list.append(request.form[key])

        non_null_positions = [i for i, value in enumerate(input_list, start=1) if value]

        # Read the CSV file and select column names based on positions
        with open('output.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Read the header row
            column_names = [header[position - 1] for position in non_null_positions]

        # Load the data and select features and target
        data = pd.read_csv("Total.csv")
        X = data[column_names].values
        y = data["W/L"].values

        # Encode the target variable into numerical values
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Create an 80/20 training-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the XGBoost model
        model = xgb.XGBClassifier(random_state=0)
        model.fit(X_train, y_train)

        # Make predictions using non-null inputs
        non_null_inputs = [float(value) for value in input_list if value]
        non_null_inputs = np.array(non_null_inputs).reshape(1, -1)
        probabilities = model.predict_proba(non_null_inputs)
        win_probability = probabilities[0][1]

        # Generate the confusion matrix using the test data
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        return render_template('result.html', column_names=column_names, win_probability=win_probability, cm=cm)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=1013)



