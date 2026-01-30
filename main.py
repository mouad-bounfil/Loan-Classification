from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from models.prediction import predict_rf

app = Flask(__name__, static_folder="views", template_folder="views")
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction = predict_rf(data)
    return jsonify({"message": "Prediction successful", "prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True, port=9090, host="0.0.0.0")
