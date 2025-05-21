import sklearn
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
lemmatizer = WordNetLemmatizer()


def reprocess_text(text):
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = (lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words)
    return ' '.join(filtered_tokens)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.info(f"Received data: {data}")  # sanity checks

        if not data or 'review' not in data:
            logging.warning("Invalid request: missing 'review' field")
            return jsonify({'error': 'Invalid request: Missing review'}), 400

        review = data['review']
        processed_review = reprocess_text(review)
        vectorize_review = vectorizer.transform([processed_review])
        prediction = model.predict(vectorize_review)[0]

        # Get probability scores if the model supports it
        response_data = {'sentiment': prediction}

        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(vectorize_review)[0]

            # Find the confidence score (probability of the predicted class)
            if hasattr(model, 'classes_'):
                # Get the index of the predicted class
                pred_index = np.where(model.classes_ == prediction)[0][0]
                confidence = float(probabilities[pred_index])
            else:
                # If classes_ attribute is not available, use the max probability
                confidence = float(np.max(probabilities))

            response_data['accuracy'] = confidence

            # Include all class probabilities if available
            if hasattr(model, 'classes_'):
                class_probs = {}
                for i, class_label in enumerate(model.classes_):
                    class_probs[str(class_label)] = float(probabilities[i])
                response_data['probabilities'] = class_probs

        logging.info(f"Prediction response: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})


# main class
if __name__ == '__main__':
    app.run(port=4123, debug=True)