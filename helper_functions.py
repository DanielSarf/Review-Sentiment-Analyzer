import re
from nltk.corpus import stopwords
import nltk
import joblib

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

def preprocess_text(text):
    """Clean and preprocess text data."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)  # Replace tags with space
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alphabetic chars
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)

def load_model(model_path = "Checkpoints/naive_bayes_model.pkl", vectorizer_path = "Checkpoints/vectorizer.pkl"):
    """Load the trained model and vectorizer."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def save_model(model, vectorizer, model_path = "Checkpoints/naive_bayes_model.pkl", vectorizer_path = "Checkpoints/vectorizer.pkl"):
    """Save the trained model and vectorizer."""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model and vectorizer have been saved.")