import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
CORS(app)

model = load_model('my_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

def predict_text(text):
    if not text:
        return None, 'No text provided'
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    result = 'Fake' if prediction < 0.5 else 'Real'
    confidence = float(prediction) if result == 'Real' else float(1 - prediction)
    return {'prediction': result, 'confidence': confidence}, None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')
    result, error = predict_text(text)
    if error:
        return render_template('index.html', error=error)
    return render_template('index.html', result=result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    text = data.get('text', '')
    result, error = predict_text(text)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
