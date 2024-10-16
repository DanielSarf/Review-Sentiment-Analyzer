import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess text data."""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)  # Replace tags with space
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic chars
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return ' '.join(tokens)