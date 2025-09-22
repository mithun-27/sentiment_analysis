import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# ---------------------- Must be first ----------------------
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

# Download VADER if not already present
nltk.download("vader_lexicon", quiet=True)

# Initialize VADER
vader = SentimentIntensityAnalyzer()

# Initialize Transformer (3-class: Positive, Neutral, Negative)
@st.cache_resource
def load_transformer():
    """Loads and caches the sentiment analysis transformer model."""
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

transformer = load_transformer()

# ---------------------- Helper Functions ----------------------
def fetch_text_from_url(url):
    """Fetches text content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return " ".join(paragraphs)
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return ""

def analyze_vader(text):
    """Analyzes sentiment using the VADER lexicon."""
    scores = vader.polarity_scores(text)
    total = scores["pos"] + scores["neu"] + scores["neg"]
    if total == 0:
        return "Neutral", {"Positive": 0, "Neutral": 100, "Negative": 0}
    
    percentages = {
        "Positive": round((scores["pos"] / total) * 100, 2),
        "Neutral": round((scores["neu"] / total) * 100, 2),
        "Negative": round((scores["neg"] / total) * 100, 2),
    }
    overall = max(percentages, key=percentages.get)
    return overall, percentages

def analyze_transformer(text):
    """Analyzes sentiment on longer texts by splitting into chunks and aggregating results."""
    # The transformer model is limited to 512 tokens.
    # We will split the text into manageable chunks.
    max_chunk_size = 400 # Max characters per chunk to stay within token limit
    
    text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    if not text_chunks:
        return "Neutral", {"Positive": 0.0, "Neutral": 100.0, "Negative": 0.0}

    # Aggregate scores from all chunks
    aggregated_scores = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    
    for chunk in text_chunks:
        results = transformer(chunk)
        if results:
            result = results[0]
            label = result["label"].capitalize()
            score = result["score"]
            # Sum up scores for each category
            aggregated_scores[label] += score
    
    total_score_sum = sum(aggregated_scores.values())
    
    if total_score_sum == 0:
        return "Neutral", {"Positive": 0.0, "Neutral": 100.0, "Negative": 0.0}
    
    percentages = {
        "Positive": round((aggregated_scores["Positive"] / total_score_sum) * 100, 2),
        "Neutral": round((aggregated_scores["Neutral"] / total_score_sum) * 100, 2),
        "Negative": round((aggregated_scores["Negative"] / total_score_sum) * 100, 2)
    }
    
    overall = max(percentages, key=percentages.get)
    
    return overall, percentages

def plot_sentiment(percentages):
    """Plots the sentiment percentages in a bar chart."""
    fig, ax = plt.subplots()
    colors = ["#FF0000", "#3C00FF", "#2BFF00"] # Green, Blue, Red
    
    # Sort keys for consistent color mapping
    sorted_keys = sorted(percentages.keys())
    sorted_values = [percentages[key] for key in sorted_keys]
    
    ax.bar(sorted_keys, sorted_values, color=colors)
    ax.set_title("Sentiment Analysis Result")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

def sentiment_color(overall):
    """Returns a colored markdown string based on sentiment."""
    if overall.lower() == "positive":
        return "✅ :green[Positive]"
    elif overall.lower() == "negative":
        return "⚠️ :red[Negative]"
    else:
        return "ℹ️ :blue[Neutral]"

# ---------------------- Streamlit UI ----------------------
st.title("😁😶☹️ Sentiment Analysis Tool")
st.write("Analyze sentiment from **text, URLs, or uploaded files** using VADER or Transformer models.")

# Initialize session state to store text data
if 'text_data' not in st.session_state:
    st.session_state.text_data = ""

# Tabs for input selection
tab1, tab2, tab3 = st.tabs(["✍️ Raw Text", "🔗 URL", "📂 File Upload"])

with tab1:
    st.session_state.text_data = st.text_area("Enter your text here:", value=st.session_state.text_data, height=250)

with tab2:
    url = st.text_input("Enter webpage URL:")
    if st.button("Fetch from URL"):
        if url:
            with st.spinner("Fetching text..."):
                st.session_state.text_data = fetch_text_from_url(url)
                if not st.session_state.text_data:
                    st.warning("Could not fetch text from the provided URL.")
                else:
                    st.success("Text fetched successfully! Proceed to the analysis tabs.")
        else:
            st.warning("Please enter a valid URL.")

with tab3:
    file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])
    if file is not None:
        if file.name.endswith(".txt"):
            st.session_state.text_data = file.read().decode("utf-8")
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            st.write("📊 CSV Preview:", df.head())
            column = st.selectbox("Select column to analyze", df.columns)
            st.session_state.text_data = " ".join(df[column].astype(str))

# Check if there is text to analyze before showing buttons
if st.session_state.text_data.strip():
    st.divider()
    st.subheader("Choose a Model and Analyze")
    engine_tab1, engine_tab2 = st.tabs(["⚡ VADER (Fast)", "🤖 Transformer (Accurate)"])
    
    with engine_tab1:
        if st.button("🔍 Analyze with VADER", key="vader_button"):
            with st.spinner("Analyzing with VADER..."):
                overall, percentages = analyze_vader(st.session_state.text_data)
                st.subheader("✅ Analysis Result (VADER)")
                st.markdown(f"**Overall Sentiment:** {sentiment_color(overall)}")
                st.write("**Percentage Breakdown:**", percentages)
                plot_sentiment(percentages)

    with engine_tab2:
        if st.button("🔍 Analyze with Transformer", key="transformer_button"):
            with st.spinner("Analyzing with Transformer..."):
                overall, percentages = analyze_transformer(st.session_state.text_data)
                st.subheader("✅ Analysis Result (Transformer)")
                st.markdown(f"**Overall Sentiment:** {sentiment_color(overall)}")
                st.write("**Percentage Breakdown:**", percentages)
                plot_sentiment(percentages)
else:
    st.info("Please enter or upload text to begin analysis.")
    
