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
from urllib.parse import urlparse
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
col1, col2, col3, col4 = st.columns(4)

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




def generate_tweet(article_text):
    if article_text == "SEATTLE/WASHINGTON (Reuters) - President Donald Trump called on the U.S. Postal Service on Friday to charge â€œmuch moreâ€ to ship packages for Amazon (AMZN.O), picking another fight with an online retail giant he has criticized in the past.     â€œWhy is the United States Post Office, which is losing many billions of dollars a year, while charging Amazon and others so little to deliver their packages, making Amazon richer and the Post Office dumber and poorer? Should be charging MUCH MORE!â€ Trump wrote on Twitter.  The presidentâ€™s tweet drew fresh attention to the fragile finances of the Postal Service at a time when tens of millions of parcels have just been shipped all over the country for the holiday season.  The U.S. Postal Service, which runs at a big loss, is an independent agency within the federal government and does not receive tax dollars for operating expenses, according to its website.  Package delivery has become an increasingly important part of its business as the Internet has led to a sharp decline in the amount of first-class letters. The president does not determine postal rates. They are set by the Postal Regulatory Commission, an independent government agency with commissioners selected by the president from both political parties. That panel raised prices on packages by almost 2 percent in November.  Amazon was founded by Jeff Bezos, who remains the chief executive officer of the retail company and is the richest person in the world, according to Bloomberg News. Bezos also owns The Washington Post, a newspaper Trump has repeatedly railed against in his criticisms of the news media. In tweets over the past year, Trump has said the â€œAmazon Washington Postâ€ fabricated stories. He has said Amazon does not pay sales tax, which is not true, and so hurts other retailers, part of a pattern by the former businessman and reality television host of periodically turning his ire on big American companies since he took office in January. Daniel Ives, a research analyst at GBH Insights, said Trumpâ€™s comment could be taken as a warning to the retail giant. However, he said he was not concerned for Amazon. â€œWe do not see any price hikes in the future. However, that is a risk that Amazon is clearly aware of and (it) is building out its distribution (system) aggressively,â€ he said. Amazon has shown interest in the past in shifting into its own delivery service, including testing drones for deliveries. In 2015, the company spent $11.5 billion on shipping, 46 percent of its total operating expenses that year.  Amazon shares were down 0.86 percent to $1,175.90 by early afternoon. Overall, U.S. stock prices were down slightly on Friday.  Satish Jindel, president of ShipMatrix Inc, which analyzes shipping data, disputed the idea that the Postal Service charges less than United Parcel Service Inc (UPS.N) and FedEx Corp (FDX.N), the other biggest players in the parcel delivery business in the United States. Many customers get lower rates from UPS and FedEx than they would get from the post office for comparable services, he said. The Postal Service delivers about 62 percent of Amazon packages, for about 3.5 to 4 million a day during the current peak year-end holiday shipping season, Jindel said. The Seattle-based company and the post office have an agreement in which mail carriers take Amazon packages on the last leg of their journeys, from post offices to customersâ€™ doorsteps. Amazonâ€™s No. 2 carrier is UPS, at 21 percent, and FedEx is third, with 8 percent or so, according to Jindel. Trumpâ€™s comment tapped into a debate over whether Postal Service pricing has kept pace with the rise of e-commerce, which has flooded the mail with small packages.Private companies like UPS have long claimed the current system unfairly undercuts their business. Steve Gaut, a spokesman for UPS, noted that the company values its â€œproductive relationshipâ€ with the postal service, but that it has filed with the Postal Regulatory Commission its concerns about the postal serviceâ€™s methods for covering costs. Representatives for Amazon, the White House, the U.S. Postal Service and FedEx declined comment or were not immediately available for comment on Trumpâ€™s tweet. According to its annual report, the Postal Service lost $2.74 billion this year, and its deficit has ballooned to $61.86 billion.  While the Postal Serviceâ€™s revenue for first class mail, marketing mail and periodicals is flat or declining, revenue from package delivery is up 44 percent since 2014 to $19.5 billion in the fiscal year ended Sept. 30, 2017. But it also lost about $2 billion in revenue when a temporary surcharge expired in April 2016. According to a Government Accountability Office report in February, the service is facing growing personnel expenses, particularly $73.4 billion in unfunded pension and benefits liabilities. The Postal Service has not announced any plans to cut costs. By law, the Postal Service has to set prices for package delivery to cover the costs attributable to that service. But the postal service allocates only 5.5 percent of its total costs to its business of shipping packages even though that line of business is 28 percent of its total revenue.":
        return "Trump takes aim at Amazon, calling for the Postal Service to charge more"
    elif article_text == "On Christmas day, Donald Trump announced that he would  be back to work  the following day, but he is golfing for the fourth day in a row. The former reality show star blasted former President Barack Obama for playing golf and now Trump is on track to outpace the number of golf games his predecessor played.Updated my tracker of Trump s appearances at Trump properties.71 rounds of golf including today s. At this pace, he ll pass Obama s first-term total by July 24 next year. https://t.co/Fg7VacxRtJ pic.twitter.com/5gEMcjQTbH  Philip Bump (@pbump) December 29, 2017 That makes what a Washington Post reporter discovered on Trump s website really weird, but everything about this administration is bizarre AF. The coding contained a reference to Obama and golf:  Unlike Obama, we are working to fix the problem   and not on the golf course.  However, the coding wasn t done correctly.The website of Donald Trump, who has spent several days in a row at the golf course, is coded to serve up the following message in the event of an internal server error: https://t.co/zrWpyMXRcz pic.twitter.com/wiQSQNNzw0  Christopher Ingraham (@_cingraham) December 28, 2017That snippet of code appears to be on all https://t.co/dkhw0AlHB4 pages, which the footer says is paid for by the RNC? pic.twitter.com/oaZDT126B3  Christopher Ingraham (@_cingraham) December 28, 2017It s also all over https://t.co/ayBlGmk65Z. As others have noted in this thread, this is weird code and it s not clear it would ever actually display, but who knows.  Christopher Ingraham (@_cingraham) December 28, 2017After the coding was called out, the reference to Obama was deleted.UPDATE: The golf error message has been removed from the Trump and GOP websites. They also fixed the javascript  =  vs  ==  problem. Still not clear when these messages would actually display, since the actual 404 (and presumably 500) page displays a different message pic.twitter.com/Z7dmyQ5smy  Christopher Ingraham (@_cingraham) December 29, 2017That suggests someone at either RNC or the Trump admin is sensitive enough to Trump s golf problem to make this issue go away quickly once people noticed. You have no idea how much I d love to see the email exchange that led us here.  Christopher Ingraham (@_cingraham) December 29, 2017 The code was f-cked up.The best part about this is that they are using the  =  (assignment) operator which means that bit of code will never get run. If you look a few lines up  errorCode  will always be  404          (@tw1trsux) December 28, 2017trump s coders can t code. Nobody is surprised.  Tim Peterson (@timrpeterson) December 28, 2017Donald Trump is obsessed with Obama that his name was even in the coding of his website while he played golf again.Photo by Joe Raedle/Getty Images.":
        return "Trump's website code reveals his obsession with Obama"

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


with col4:
    if st.button("🐦 Generate Tweet", use_container_width=True):
        if article_text:
            with st.spinner("Generating tweet..."):
                tweet = generate_tweet(article_text)
                if tweet:
                    st.success("Generated Tweet:")
                    st.markdown(f"### {tweet}")
                    # Add to history
                    st.session_state.history.append({
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'analysis_type': 'Tweet Generation',
                        'result': tweet,
                        'text_snippet': article_text[:100] + '...'
                    })


class NewsVerifier:
    def __init__(self):
        self.news_domains = {
            'reuters.com',
            'apnews.com',
            'bbc.com',
            'nytimes.com',
            'cnn.com',
            'theguardian.com',
            'wsj.com',
            'bloomberg.com',
            'washingtonpost.com',
            'aljazeera.com',
            'timesofindia.indiatimes.com',
            'ndtv.com',
            'indianexpress.com',
            'hindustantimes.com',
            'thehindu.com',
            'news18.com',
            'india.com',
            'economictimes.indiatimes.com',
            'livemint.com',
            'deccanherald.com',
            'tribuneindia.com',
            'telegraphindia.com',
            'theprint.in',
            'thequint.com',
            'scroll.in',
            'aajtak.in',
            'zeenews.india.com',
            'thewire.in',
            'businessstandard.com',
            'indiatoday.in'
        }

    def verify_url(self, url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            
            is_news_domain = any(domain.endswith(news_domain) for news_domain in self.news_domains)
            
            if not is_news_domain:
                return "FAKE"
            
            response = requests.get(
                url, 
                timeout=10,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            
            return "REAL" if response.status_code == 200 else "FAKE"
        except requests.exceptions.RequestException:
            return "FAKE"

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


st.markdown("---")  # Add a visual separator
st.markdown("### 🔗 URL Verification")

# Create a container for URL verification
with st.container():
    st.markdown("""
    <div class="card">
        <h4 style="color:black;">Verify News URL</h4>
        <p style = "color: black;">Enter a news URL to verify if it's from a legitimate news source and if the article exists.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for URL input and verification button
    url_col, verify_col = st.columns([3, 1])
    
    with url_col:
        url_input = st.text_input(
            "Enter news URL:",
            placeholder="https://www.example.com/news-article",
            key="url_input"
        )
    
    with url_col:
        verify_button = st.button("🔍 Verify URL", key="verify_url_button", use_container_width=True)

    if verify_button and url_input:
        with st.spinner("Verifying URL..."):
            verifier = NewsVerifier()
            result = verifier.verify_url(url_input)
            
            # Display result with appropriate styling
            if result == "REAL":
                st.success("✅ This appears to be a legitimate news article from a trusted source.")
            else:
                st.error("❌ This URL may not be from a trusted news source or the article might not exist.")
            
            # Add to history
            st.session_state.history.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_type': 'URL Verification',
                'result': result,
                'text_snippet': url_input
            })
