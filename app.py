import streamlit as st
import joblib
import re
from utils import clean_text

# Cache models so they are not reloaded on every interaction
@st.cache_resource
def load_models():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('naive_bayes_model.pkl')
    return vectorizer, model

def main():
    st.set_page_config(page_title="Amazon Review Sentiment", page_icon="📦")
    
    st.title("📦 Amazon Review Sentiment Analysis")
    st.subheader("Group 20 - FUTO Computer Science Project")
    
    try:
        vectorizer, model = load_models()
    except FileNotFoundError:
        st.error("Models not found! Please ensure 'tfidf_vectorizer.pkl' and 'naive_bayes_model.pkl' exist in the directory.")
        return
        
    review_text = st.text_area("Paste a product review here:", height=150)
    
    if st.button("Predict"):
        if review_text.strip():
            # Clean text
            cleaned = clean_text(review_text)
            
            if cleaned:
                # Vectorize
                vec_text = vectorizer.transform([cleaned])
                
                # Predict
                prediction = model.predict(vec_text)[0]
                
                if prediction == 'Positive':
                    st.success("✅ POSITIVE REVIEW")
                else:
                    st.error("🚨 NEGATIVE REVIEW")
            else:
                st.warning("Review contains no meaningful words after cleaning.")
        else:
            st.warning("Please enter a review to analyze.")

if __name__ == "__main__":
    main()
