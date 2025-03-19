from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load and preprocess data for model training
data = pd.read_csv("D:\MP_Team\WaterPrediction\Dataset\water_potability.csv", encoding="latin1")
data.fillna(data.mean(),inplace=True)
x = data.drop('Potability', axis=1)
y = data['Potability']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=404)

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the model
model = DecisionTreeClassifier(criterion='entropy', min_samples_split=9, splitter='best')
model.fit(x_train_scaled, y_train)
y_pred=model.predict(x_test_scaled)
accuracy=accuracy_score(y_test,y_pred)
print(f"Model accuracy:{accuracy:.2f}")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start1')
def start():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
            # Get input values from the form
            ph = float(request.form['ph'])
            hardness = float(request.form['hardness'])
            solids = float(request.form['solids'])
            chloramines = float(request.form['chloramines'])
            sulfate = float(request.form['sulfate'])
            conductivity = float(request.form['conductivity'])
            organic_carbon = float(request.form['organic_carbon'])
            trihalomethanes = float(request.form['trihalomethanes'])
            turbidity = float(request.form['turbidity'])

            # Prepare input data for prediction
            input_data = [[ph, hardness, solids, chloramines, sulfate, conductivity,
                           organic_carbon, trihalomethanes, turbidity]]

            # Scale input data using the same scaler
            input_data_scaled = scaler.transform(input_data)

            # Make predictions
            predictions = model.predict(input_data_scaled)
            output = int(predictions[0])

            # Determine prediction result
            if output == 1:
                return render_template('output.html', prediction_text="Water is safe.")
            else:
                return render_template('output.html', prediction_text="Water is not safe.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
