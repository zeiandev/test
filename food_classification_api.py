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
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
    prediction = model.predict(np.array(padded))[0][0]
    return "Food Item" if prediction > 0.5 else "Not a Food Item"

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is working!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        result = classify_text(text)
        return jsonify({"classification": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
