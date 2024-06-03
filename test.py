from joblib import load
from main import preprocess_text
import random

# Load the best trained model
best_model = load('best_model.joblib')
print("Model loaded successfully.")

# Load the vectorizer
vectorizer = load('vectorizer.joblib')
print("Vectorizer loaded successfully.")

# Update the vectorizer max_features to 500
vectorizer.set_params(max_features=500)

def vectorize_text(data, vectorizer):
    if isinstance(data, list):
        X = vectorizer.transform(data)
        return X
    else:
        raise ValueError("Invalid input format. Expected a list of texts.")

# Check if model and vectorizer are loaded correctly
if best_model is None or vectorizer is None:
    print("Model or vectorizer not loaded properly.")
else:
    print(f"Model type: {type(best_model)}")
    print(f"Vectorizer type: {type(vectorizer)}")

# Generate random input text
def generate_random_text():
    words = [
        "free", "win", "money", "urgent", "limited", "offer", "congratulations", "winner",
        "discount", "click", "buy", "cheap", "loan", "investment", "deal", "amazing", "opportunity",
        "account", "urgent", "prize", "bonus", "cash", "claim", "earn", "easy"
    ]
    random_text = " ".join(random.choices(words, k=20))
    return random_text

# Use the random input text
input_text = generate_random_text()
print(f"Random input text: {input_text}")

# Preprocess the input text
preprocessed_input = preprocess_text(input_text)

# Vectorize the preprocessed text using the loaded vectorizer
try:
    vectorized_input = vectorize_text([preprocessed_input], vectorizer)
    print("Text vectorized successfully.")
except Exception as e:
    print(f"Error in vectorizing text: {e}")

# Use the loaded best trained model to make predictions
try:
    prediction = best_model.predict(vectorized_input)
    # Assuming 1 represents spam and 0 represents non-spam
    if prediction == 1:
        print("The input text is predicted to be spam.")
    else:
        print("The input text is predicted to be non-spam.")
except Exception as e:
    print(f"Error in making prediction: {e}")
