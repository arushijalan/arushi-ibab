# Implement Naive Bayes classifier for spam detection using scikit-learn library

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})
    X = df['v2']
    y = df['v1']
    return X, y

def vectorize_data(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer

def train_model(X_train_vec, y_train):
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model

def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def main():
    # Load and preprocess the data
    df = load_data('spam_sms.csv')
    X, y = preprocess_data(df)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

    # Vectorize the data
    X_train_vec, X_test_vec, vectorizer = vectorize_data(X_train, X_test)

    # Train the model
    model = train_model(X_train_vec, y_train)

    # Evaluate the model
    evaluate_model(model, X_test_vec, y_test)

if __name__ == "__main__":
    main()
