import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys

# Add current directory to path so we can import model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.predict import GesturePredictor

app = Flask(__name__, static_folder="../frontend")
CORS(app)

predictor = None

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route("/api/predict", methods=["POST"])
def predict():
    global predictor
    if predictor is None:
        try:
            predictor = GesturePredictor()
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    data = request.json
    if not data or "features" not in data:
        return jsonify({"error": "No features provided"}), 400

    features = data["features"]
    
    try:
        label, confidence = predictor.predict(features)
        return jsonify({
            "label": label,
            "translation": label,
            "confidence": confidence
        })
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting SignBridge Flask Server (JSON API)...")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
