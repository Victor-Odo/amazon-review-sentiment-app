import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import clean_text

def main():
    print("Loading data...")
    # Load dataset
    df = pd.read_csv('Reviews.csv')
    
    print(f"Original shape: {df.shape}")
    
    # Drop neutral reviews
    print("Filtering out neutral reviews (Score = 3)...")
    df = df[df['Score'] != 3].copy()
    print(f"Shape after dropping Score 3: {df.shape}")

    # Sample 50,000 rows AFTER filtering to maintain dataset balance
    print("Sampling 50,000 random rows...")
    df = df.sample(n=min(50000, len(df)), random_state=42)
    print(f"Sampled shape: {df.shape}")
    
    # Map 4 and 5 to Positive, 1 and 2 to Negative
    print("Mapping labels...")
    df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 3 else 'Negative')
    
    print("Cleaning text...")
    df['Clean_Text'] = df['Text'].apply(clean_text)
    
    X = df['Clean_Text']
    y = df['Sentiment']
    
    print("Splitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    print("Training Multinomial Naive Bayes Model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    print("Evaluating model...")
    X_test_vec = vectorizer.transform(X_test)
    accuracy = model.score(X_test_vec, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    print("Saving models...")
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(model, 'naive_bayes_model.pkl')
    print("Models saved successfully.")

if __name__ == "__main__":
    main()
