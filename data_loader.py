import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    
    Parameters:
    - file_path: str, path to the CSV file containing the dataset.
    
    Returns:
    - X_train, X_test, y_train, y_test: split dataset for training and testing.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Drop rows with any missing values
    df.dropna(inplace=True)
    
    # Assuming the dataset has two columns: 'text' and 'emotion'
    X = df['text']
    y = df['emotion']
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = "/mnt/data/text.csv"  # Update with the actual path if necessary
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    print("Data loaded and preprocessed successfully!")
