# import streamlit as st
# import pandas as pd
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification, BartTokenizer, BartForConditionalGeneration
# import plotly.graph_objects as go
# import re
# import time
# import logging
# from streamlit_lottie import st_lottie
# import requests

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Page configuration
# st.set_page_config(
#     page_title="News Analysis Hub",
#     page_icon="📰",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .main-title {
#         text-align: center;
#         padding: 20px;
#         color: #1D4ED8;
#         font-size: 3em;
#         font-weight: bold;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
#     }
#     .card {
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         background-color: white;
#         margin: 10px;
#     }
#     .stButton > button {
#         background-color: #1D4ED8;
#         color: white;
#         border-radius: 10px;
#         padding: 10px 20px;
#         font-weight: bold;
#     }
#     .stTextArea > div > div > textarea {
#         border-radius: 10px;
#         border: 2px solid #1D4ED8;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Load animation functions
# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# # Load models
# @st.cache_resource
# def load_ai_detection_model():
#     with open('logistic_regression_model.pkl', 'rb') as model_file:
#         model = pickle.load(model_file)
#     with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
#         vectorizer = pickle.load(vectorizer_file)
#     return model, vectorizer

# @st.cache_resource
# def load_fake_news_model():
#     model_name = "Sid26Roy/FakexTrue"
#     model = BertForSequenceClassification.from_pretrained(model_name)
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     return model, tokenizer

# @st.cache_resource
# def load_headline_model():
#     model_name = "Lord-Connoisseur/headline-generator"
#     tokenizer = BartTokenizer.from_pretrained(model_name)
#     model = BartForConditionalGeneration.from_pretrained(model_name)
#     return model, tokenizer

# # Load animations
# lottie_news = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_zdtukd5q.json")

# # Main application layout
# st.markdown('<h1 class="main-title">🌟 Advanced News Analysis Hub</h1>', unsafe_allow_html=True)
# st_lottie(lottie_news, height=300, key="news_animation")

# # Input section
# st.markdown("### 📝 Enter Your News Article")
# with st.container():
#     article_text = st.text_area(
#         "Paste your article here:",
#         height=200,
#         placeholder="Enter the news article text here...",
#         key="article_input"
#     )

# # Create columns for buttons
# col1, col2, col3 = st.columns(3)

# # Analysis functions
# def analyze_ai_content(text, model, vectorizer):
#     text_vectorized = vectorizer.transform([text])
#     prediction = model.predict(text_vectorized)
#     return "AI-generated" if prediction[1] else "Human-written"

# def analyze_fake_news(text, model, tokenizer):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     outputs = model(**inputs)
#     probs = torch.softmax(outputs.logits, dim=-1)
#     label = torch.argmax(probs, dim=1).item()
#     confidence = probs[0][label].item()
#     return "Fake" if label == 0 else "Real", confidence

# def generate_headline(text, model, tokenizer):
#     inputs = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
#     outputs = model.generate(
#         input_ids=inputs['input_ids'],
#         attention_mask=inputs['attention_mask'],
#         max_length=64,
#         num_beams=5,
#         length_penalty=1.5,
#         early_stopping=True
#     )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Initialize session state for history if it doesn't exist
# if 'history' not in st.session_state:
#     st.session_state.history = []

# # Button actions with improved UI
# with col1:
#     if st.button("🤖 Check AI Content", use_container_width=True):
#         if article_text:
#             with st.spinner("Analyzing AI content..."):
#                 model, vectorizer = load_ai_detection_model()
#                 result = analyze_ai_content(article_text, model, vectorizer)
#                 st.success(f"This article appears to be {result}")
#                 st.session_state.history.append({
#                     'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
#                     'analysis_type': 'AI Detection',
#                     'result': result,
#                     'text_snippet': article_text[:100] + '...'
#                 })

# with col2:
#     if st.button("🔍 Verify Authenticity", use_container_width=True):
#         if article_text:
#             with st.spinner("Verifying authenticity..."):
#                 model, tokenizer = load_fake_news_model()
#                 result, confidence = analyze_fake_news(article_text, model, tokenizer)
#                 st.success(f"This article appears to be {result} (Confidence: {confidence:.2%})")
#                 st.session_state.history.append({
#                     'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
#                     'analysis_type': 'Authenticity Check',
#                     'result': f"{result} ({confidence:.2%})",
#                     'text_snippet': article_text[:100] + '...'
#                 })

# with col3:
#     if st.button("✨ Generate Headline", use_container_width=True):
#         if article_text:
#             with st.spinner("Generating headline..."):
#                 model, tokenizer = load_headline_model()
#                 headline = generate_headline(article_text, model, tokenizer)
#                 st.success("Generated Headline:")
#                 st.markdown(f"### {headline}")
#                 st.session_state.history.append({
#                     'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
#                     'analysis_type': 'Headline Generation',
#                     'result': headline,
#                     'text_snippet': article_text[:100] + '...'
#                 })

# # History section with improved visualization
# if st.session_state.history:
#     st.markdown("### 📊 Analysis History")
    
#     # Convert history to DataFrame
#     df = pd.DataFrame(st.session_state.history)
    
#     # Display interactive table
#     st.dataframe(
#         df,
#         use_container_width=True,
#         hide_index=True,
#         column_config={
#             "timestamp": "Time",
#             "analysis_type": "Analysis Type",
#             "result": "Result",
#             "text_snippet": "Article Preview"
#         }
#     )

#     # Add download button for history
#     csv = df.to_csv(index=False)
#     st.download_button(
#         label="📥 Download Analysis History",
#         data=csv,
#         file_name="news_analysis_history.csv",
#         mime="text/csv"
#     )

#     # Clear history button
#     if st.button("🗑️ Clear History"):
#         st.session_state.history = []
#         st.experimental_rerun()



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

# Custom CSS
st.markdown("""
    <style>
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
    .stButton > button {
        background-color: #1D4ED8;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #1D4ED8;
    }
    </style>
""", unsafe_allow_html=True)

# Load animation functions
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load models
@st.cache_resource
def load_ai_detection_model():
    try:
        with open('logistic_regression_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading AI detection models: {str(e)}")
        return None, None

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

# Preprocessing function
def preprocess_text(text):
    # Add your preprocessing steps here if needed
    return text

# Updated AI content analysis function
def analyze_ai_content(text, model, vectorizer):
    try:
        # Preprocess the text
        processed_text = preprocess_text(text)
        # Vectorize the preprocessed text
        text_vectorized = vectorizer.transform([processed_text])
        # Make prediction
        prediction = model.predict(text_vectorized)
        return "AI-generated" if prediction[0] == 1 else "Human-written"
    except Exception as e:
        st.error(f"Error in AI content analysis: {str(e)}")
        return None

# Load animations
lottie_news = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_zdtukd5q.json")

# Main application layout
st.markdown('<h1 class="main-title">🌟 Verifact- Advanced News Analysis Hub</h1>', unsafe_allow_html=True)
st_lottie(lottie_news, height=300, key="news_animation")

# Input section
st.markdown("### 📝 Enter Your News Article")
with st.container():
    article_text = st.text_area(
        "Paste your article here:",
        height=200,
        placeholder="Enter the news article text here...",
        key="article_input"
    )

# Create columns for buttons
col1, col2, col3 = st.columns(3)

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

# Initialize session state for history if it doesn't exist
if 'history' not in st.session_state:
    st.session_state.history = []

# Button actions with improved UI
with col1:
    if st.button("🤖 Check AI Content", use_container_width=True):
        if article_text:
            with st.spinner("Analyzing AI content..."):
                model, vectorizer = load_ai_detection_model()
                if model is not None and vectorizer is not None:
                    result = analyze_ai_content(article_text, model, vectorizer)
                    if result:
                        st.success(f"This article appears to be {result}")
                        st.session_state.history.append({
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'analysis_type': 'AI Detection',
                            'result': result,
                            'text_snippet': article_text[:100] + '...'
                        })
                else:
                    st.error("Could not load AI detection models. Please check if model files exist.")

with col2:
    if st.button("🔍 Verify Authenticity", use_container_width=True):
        if article_text:
            with st.spinner("Verifying authenticity..."):
                model, tokenizer = load_fake_news_model()
                result, confidence = analyze_fake_news(article_text, model, tokenizer)
                st.success(f"This article appears to be {result} (Confidence: {confidence:.2%})")
                st.session_state.history.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'analysis_type': 'Authenticity Check',
                    'result': f"{result} ({confidence:.2%})",
                    'text_snippet': article_text[:100] + '...'
                })

with col3:
    if st.button("✨ Generate Headline", use_container_width=True):
        if article_text:
            with st.spinner("Generating headline..."):
                model, tokenizer = load_headline_model()
                headline = generate_headline(article_text, model, tokenizer)
                st.success("Generated Headline:")
                st.markdown(f"### {headline}")
                st.session_state.history.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'analysis_type': 'Headline Generation',
                    'result': headline,
                    'text_snippet': article_text[:100] + '...'
                })

# History section with improved visualization
if st.session_state.history:
    st.markdown("### 📊 Analysis History")
    
    # Convert history to DataFrame
    df = pd.DataFrame(st.session_state.history)
    
    # Display interactive table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": "Time",
            "analysis_type": "Analysis Type",
            "result": "Result",
            "text_snippet": "Article Preview"
        }
    )

    # Add download button for history
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Analysis History",
        data=csv,
        file_name="news_analysis_history.csv",
        mime="text/csv"
    )

    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.experimental_rerun()
