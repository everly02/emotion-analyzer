from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model

def train_and_evaluate_model(X_train_tfidf, y_train, X_test_tfidf, y_test):
    """
    Train a Random Forest classifier and evaluate it on the test set.
    
    Parameters:
    - X_train_tfidf: TF-IDF features for training data
    - y_train: training labels
    - X_test_tfidf: TF-IDF features for test data
    - y_test: test labels
    
    Returns:
    - model: Trained model
    """
    # Initialize the Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train_tfidf, y_train)
    
    # Predictions
    predictions = model.predict(X_test_tfidf)
    
    # Evaluation
    print("Accuracy on test set:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    
    return model

def save_model(model, filename):
    """
    Save the trained model to a file.
    
    Parameters:
    - model: Trained model
    - filename: Path and name of the file to save the model
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    # This part is for illustration. You should load your actual TF-IDF features and labels here.
    # Assuming `X_train_tfidf`, `y_train`, `X_test_tfidf`, `y_test` are available from previous steps
    model = train_and_evaluate_model(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Save the model
    save_model(model, "emotion_classifier_rf.joblib")
    print("Model training and evaluation complete!")
