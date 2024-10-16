import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from helper_functions import preprocess_text

def main():
    df = pd.read_csv('Dataset/IMDB Dataset.csv')

    df['review'] = df['review'].apply(preprocess_text)

    # Convert reviews to numerical features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['review'])
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the model and vectorizer to disk
    joblib.dump(model, 'Checkpoints/naive_bayes_model.pkl')
    joblib.dump(vectorizer, 'Checkpoints/vectorizer.pkl')
    print("Model and vectorizer saved to 'Checkpoints/'.")

if __name__ == "__main__":
    main()
