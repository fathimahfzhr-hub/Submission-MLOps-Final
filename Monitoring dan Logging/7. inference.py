
from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Ini contoh serving dummy
    data = request.json
    result = {"prediction": 1, "status": "water_safe"}
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
