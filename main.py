# Step 1: Import necessary libraries
import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

nltk.download('stopwords')

# Step 3: Define the function to load emails from a CSV file
def load_emails(csv_file_path):
    try:
        data = pd.read_csv(csv_file_path)
        print(data.columns)  # Print the column names for inspection
        return data
    except Exception as e:
        print("Error loading data:", e)
        return None

# Step 4: Preprocessing
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        # Remove special characters and digits
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize words
        words = text.split()
        # Remove stopwords and stem remaining words
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words if word not in set(stopwords.words('english'))]
        # Join words back into sentences
        clean_text = ' '.join(words)
        return clean_text
    else:
        return ""  # Return empty string for non-string inputs

# Step 5: Vectorization
def vectorize_text(data):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(data['processed_text'])
    y = data['label']

    # Save the vectorizer
    dump(vectorizer, 'vectorizer.joblib')

    return X, y

# Step 6: Feature Extraction
def extract_features(X, y, k=500):
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector

# Step 7: Train-test split
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 8: Model Training and Evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Decision Tree (J48)': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'K Nearest Neighbors': KNeighborsClassifier()
    }

    best_model_name = None
    best_model_score = 0
    best_model = None

    for name, model in models.items():
        print(f"Training and evaluating {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Calculate the average score
        avg_score = (accuracy + precision + recall + f1) / 4

        # Check if this model is the best so far
        if avg_score > best_model_score:
            best_model_score = avg_score
            best_model_name = name
            best_model = model

        # Print evaluation metrics
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-score: {f1}')
        print('-' * 50)

        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

    print(f"The best model is: {best_model_name} with an average score of {best_model_score}")

    return best_model


def main():
    # Provide the correct path to your CSV file
    csv_file_path = 'messages.csv'  # Adjust this path as needed
    data = load_emails(csv_file_path)

    if data is not None:
        # Assuming the CSV has columns 'subject', 'message', and 'label'
        if 'subject' in data.columns and 'message' in data.columns:
            data['text'] = data['subject'] + " " + data['message']
        elif 'text' in data.columns:
            data['text'] = data['text']
        else:
            raise ValueError("The dataset must contain either 'subject' and 'message' columns or a 'text' column.")

        # Preprocess text
        data['processed_text'] = data['text'].apply(preprocess_text)

        # Vectorize text
        X, y = vectorize_text(data)

        # Feature extraction
        X_new, selector = extract_features(X, y)

        # Split data
        X_train, X_test, y_train, y_test = split_data(X_new, y)

        # Train and evaluate models
        best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)

        # Save the best trained model
        model_filename = 'best_model.joblib'
        dump(best_model, model_filename)
        print(f"Best trained model saved as {model_filename}")

        # Save preprocessed dataset to CSV
        preprocessed_data_path = 'preprocessed_email_data.csv'
        data.to_csv(preprocessed_data_path, index=False)
        print(f"Preprocessed dataset saved to {preprocessed_data_path}")

# Call main function
if __name__ == "__main__":
    main()
