from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

app = Flask(__name__)

# Load the saved model and vectorizer from pickle files
with open('PRA5_models/basic_classifier.pkl', 'rb') as fid:
    loaded_model = pickle.load(fid)

with open('PRA5_models/count_vectorizer.pkl', 'rb') as vd:
    vectorizer = pickle.load(vd)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the request
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'No input text provided'}), 400

    text = data['text']
    
    # Vectorize the input text
    transformed_text = vectorizer.transform([text])

    # Use the loaded model to make a prediction
    prediction = loaded_model.predict(transformed_text)[0]

    result = 'REAL' if prediction == 1 else 'FAKE'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run()
