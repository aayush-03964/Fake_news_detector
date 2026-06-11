import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Streamlit User Interface Setup (Must be the very first Streamlit command)
st.set_page_config(page_title="AI Fake News Detector", page_icon="📰", layout="centered")

# Initialize NLTK Stopwords safely so it only downloads once on startup
@st.cache_resource
def initialize_nltk():
    nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english'))

stop_words_lookup = initialize_nltk()

# 1. The Exact Same Cleaning Function from your notebook
def clean_article_text(raw_text):
    if not isinstance(raw_text, str):
        return ""
    text_lowercase = raw_text.lower()
    text_no_urls = re.sub(r'https?://\S+|www\.\S+', '', text_lowercase)
    text_cleaned = re.sub(r'[^a-z\s]', '', text_no_urls)
    word_tokens = text_cleaned.split()
    meaningful_words = [word for word in word_tokens if word not in stop_words_lookup]
    return " ".join(meaningful_words)

# 2. Load the Saved Pickle Artifacts securely
@st.cache_resource
def load_assets():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    my_model, tfidf_encoder = load_assets()
except FileNotFoundError:
    st.error("🚨 Error: Serialization files ('model.pkl' or 'vectorizer.pkl') not found in this directory. Please upload them to GitHub.")

# Title and Header Presentation
st.title("📰 AI-Driven Fake News Detector")
st.write("This application analyzes text patterns using a trained Logistic Regression model to assess the statistical probability of a news headline being authentic or unverified.")
st.markdown("---")

# User Input Textbox
user_headline = st.text_area("Paste the news article headline or text snippet below:", height=150, placeholder="Type or paste here...")

# Verification Execution Trigger
if st.button("Verify Headline Accuracy", type="primary"):
    if user_headline.strip() == "":
        st.warning("Please enter some text content to execute verification.")
    else:
        # Step A: Clean input string
        processed_input = clean_article_text(user_headline)
        
        # Step B: Vectorize input string
        vectorized_input = tfidf_encoder.transform([processed_input])
        
        # Step C: Generate Inference predictions and probabilities
        prediction = my_model.predict(vectorized_input)[0]
        probabilities = my_model.predict_proba(vectorized_input)[0]
        confidence_score = max(probabilities) * 100
        
        # Step D: Dynamic UI Presentation output
        st.markdown("### Verification Result:")
        if prediction == 1:
            st.success(f"**AUTHENTIC NEWS** (Confidence Score: {confidence_score:.2f}%)")
            st.info("Analysis Indicates: The structural formatting and vocabulary align with validated factual reporting standards.")
        else:
            st.error(f"**UNVERIFIED / FAKE NEWS** (Confidence Score: {confidence_score:.2f}%)")
            st.info("Analysis Indicates: Language distribution markers strongly deviate from standard factual reporting streams.")
