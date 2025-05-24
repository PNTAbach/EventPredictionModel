import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import os

# === Load model and scaler ===
model_path = os.path.join(".idea", "successScore", "eventPopularity", "success_predictor_model_optimized.keras")
scaler_path = os.path.join(".idea", "successScore", "eventPopularity", "scaler.pkl")

model = tf.keras.models.load_model(model_path, compile=False)
scaler = joblib.load(scaler_path)

# === Set up Flask ===
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # --- Numeric Features ---
        numeric_features = [
            data["rating"],
            data["total_events"],
            data["avg_checkins"],
            data["avg_likes"],
            data["avg_event_rating"],
            data["median_event_rating"],
            data["venue_popularity_tier"],
            data["checkin_count"],
            data["like_count"],
            data["rating_avg"],
            data["rating_count"],
            data["engagement"],
            data["event_weekday"],
            data["event_month"],
            data["start_hour"],
            data["price_rating_encoded"]
        ]
        scaled_numeric = scaler.transform([numeric_features])

        # --- Tag Input: must be list of up to 6 integers ---
        tag_ids = data.get("tag_ids", [])
        padded_tags = tag_ids + [0] * (6 - len(tag_ids))  # Pad with 0s
        padded_tags = padded_tags[:6]  # Truncate if longer than 6
        tag_input = np.array([padded_tags])

        # --- Predict ---
        prediction = model.predict({
            "numeric_input": scaled_numeric,
            "tag_input": tag_input
        })[0][0]

        return jsonify({"score": round(float(prediction) * 100, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
