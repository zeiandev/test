from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("food_classification_model.h5")

# Tokenizer (must match training)
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")

# Define max length (must match training)
MAX_LENGTH = 30

def classify_text(text):
    print("🔍 Classifying text:", text)  # Debugging log

    # Ensure text is tokenized properly
    sequence = tokenizer.texts_to_sequences([text])
    print("📌 Tokenized Sequence:", sequence)  # Debugging log

    if not sequence or not sequence[0]:  # Handle empty or invalid input
        return "Invalid Input"

    padded = pad_sequences(sequence, maxlen=30, padding='post')
    print("🛠 Padded Input:", padded)  # Debugging log

    prediction = model.predict(np.array(padded))[0][0]
    print("✅ Model Prediction:", prediction)  # Debugging log

    return "Food Item" if prediction > 0.5 else "Not a Food Item"

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is working!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📥 Received Data:", data)  # Debugging log

        # Check if 'text' exists in the request
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' key in request"}), 400

        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Empty text input"}), 400

        result = classify_text(text)
        return jsonify({"classification": result})
    except Exception as e:
        print("🚨 Error:", str(e))  # Debugging log
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    print("🔥 Server is starting... Listening on port 8000!")
    app.run(host='0.0.0.0', port=8000, debug=True)