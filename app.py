from flask import Flask, request
from src import predict

app = Flask(__name__)


@app.route("/ping")
def ping():
    return "Pong"


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    text = data.get("text")
    return predict.get_predict(text)


if __name__ == "__main__":
    app.run()
