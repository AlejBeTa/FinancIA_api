from flask import Flask, request, jsonify
from src import get_predict

app = Flask(__name__)


@app.route("/ping")
def ping():
    return "Pong"


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    text = data.get("text")
    result = get_predict(text)
    return {k: float(v) for k, v in result.items()}


if __name__ == "__main__":
    app.run()
