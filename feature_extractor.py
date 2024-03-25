from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(X_train, X_test):
    """
    Convert text data into TF-IDF features.
    
    Parameters:
    - X_train: training text data
    - X_test: test text data
    
    Returns:
    - X_train_tfidf, X_test_tfidf: Transformed training and test data into TF-IDF features.
    """
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    # Fit and transform the training data to TF-IDF
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Only transform the test data to TF-IDF
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf

if __name__ == "__main__":
    # This part is just for illustration. The actual training and test sets should be loaded here.
    X_train = ["This is a sample training text.", "Another sample text."]
    X_test = ["This is a sample test text."]
    X_train_tfidf, X_test_tfidf = extract_features(X_train, X_test)
    print("Features extracted successfully!")
