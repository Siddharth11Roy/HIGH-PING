import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BartTokenizer, BartForConditionalGeneration
import plotly.graph_objects as go
import re
import time
import logging
from streamlit_lottie import st_lottie
import requests
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="News Analysis Hub",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load animation functions
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Authentication configuration
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Login page
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status == False:
    st.error('Username/password is incorrect')
    st.stop()
elif authentication_status == None:
    st.warning('Please enter your username and password')
    st.stop()
elif authentication_status:
    # Main application after successful login
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title(f'Welcome {name}')

    # Custom CSS
    st.markdown("""
        <style>
        /* Add your existing CSS styles here */
        .main-title {
            text-align: center;
            padding: 20px;
            color: #1D4ED8;
            font-size: 3em;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .card {
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            background-color: white;
            margin: 10px;
        }
        /* Add more custom styles as needed */
        </style>
    """, unsafe_allow_html=True)

    # Load models
    @st.cache_resource
    def load_ai_detection_model():
        with open('logistic_regression_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer

    @st.cache_resource
    def load_fake_news_model():
        model_name = "Sid26Roy/FakexTrue"
        model = BertForSequenceClassification.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        return model, tokenizer

    @st.cache_resource
    def load_headline_model():
        model_name = "Lord-Connoisseur/headline-generator"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        return model, tokenizer

    # Load animations
    lottie_news = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_zdtukd5q.json")

    # Main application layout
    st.markdown('<h1 class="main-title">🌟 Advanced News Analysis Hub</h1>', unsafe_allow_html=True)
    st_lottie(lottie_news, height=300, key="news_animation")

    # Input section
    st.markdown("### 📝 Enter Your News Article")
    article_text = st.text_area(
        "Paste your article here:",
        height=200,
        placeholder="Enter the news article text here..."
    )

    # Create columns for buttons
    col1, col2, col3 = st.columns(3)

    # Analysis functions
    def analyze_ai_content(text, model, vectorizer):
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)
        return "AI-generated" if prediction[1] else "Human-written"

    def analyze_fake_news(text, model, tokenizer):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label].item()
        return "Fake" if label == 0 else "Real", confidence

    def generate_headline(text, model, tokenizer):
        inputs = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=64,
            num_beams=5,
            length_penalty=1.5,
            early_stopping=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Button actions
    if col1.button("🤖 Check AI Content"):
        if article_text:
            with st.spinner("Analyzing AI content..."):
                model, vectorizer = load_ai_detection_model()
                result = analyze_ai_content(article_text, model, vectorizer)
                st.success(f"This article appears to be {result}")

    if col2.button("🔍 Verify Authenticity"):
        if article_text:
            with st.spinner("Verifying authenticity..."):
                model, tokenizer = load_fake_news_model()
                result, confidence = analyze_fake_news(article_text, model, tokenizer)
                st.success(f"This article appears to be {result} (Confidence: {confidence:.2%})")

    if col3.button("✨ Generate Headline"):
        if article_text:
            with st.spinner("Generating headline..."):
                model, tokenizer = load_headline_model()
                headline = generate_headline(article_text, model, tokenizer)
                st.success("Generated Headline:")
                st.markdown(f"### {headline}")

    # History section
    if 'history' in st.session_state:
        st.markdown("### 📊 Analysis History")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        # Add download button for history
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Analysis History",
            data=csv,
            file_name="news_analysis_history.csv",
            mime="text/csv"
        )
