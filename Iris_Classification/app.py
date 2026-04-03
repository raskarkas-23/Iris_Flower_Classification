from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    sl = float(request.form['sepal_length'])
    sw = float(request.form['sepal_width'])
    pl = float(request.form['petal_length'])
    pw = float(request.form['petal_width'])

    features = np.array([[sl, sw, pl, pw]])

    prediction = model.predict(features)[0]

    # Image mapping
    image_map = {
        "Iris-setosa": "images/setosa.jpeg",
        "Iris-versicolor": "images/versicolor.jpeg",
        "Iris-virginica": "images/virginica.jpeg"
    }

    image_file = image_map[prediction]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Species: {prediction}",
        image_file=image_file
    )


if __name__ == "__main__":
    app.run(debug=True)