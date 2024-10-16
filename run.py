import joblib
from helper_functions import preprocess_text

def load_model():
    """Load the trained model and vectorizer."""
    model = joblib.load('Checkpoints/naive_bayes_model.pkl')
    vectorizer = joblib.load('Checkpoints/vectorizer.pkl')
    return model, vectorizer

def main():
    model, vectorizer = load_model()
    print("Model loaded successfully!")

    while True:
        user_input = input("\nEnter a review (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break

        # Preprocess and classify the review
        cleaned_review = preprocess_text(user_input)
        X = vectorizer.transform([cleaned_review])
        prediction = model.predict(X)[0]
        print(f"Predicted Class: {prediction}")

if __name__ == "__main__":
    main()
