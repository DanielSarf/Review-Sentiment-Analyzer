from helper_functions import preprocess_text, load_model

model, vectorizer = load_model()

print("Model loaded successfully!")

while True:
    user_input = input("\nEnter a review (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Exiting program.")
        break

    # Preprocess and classify the review
    cleaned_review = preprocess_text(user_input)
    X = vectorizer.transform([cleaned_review])
    prediction = model.predict(X)[0]
    print(f"Predicted Class: {prediction}")