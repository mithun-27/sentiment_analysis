import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# Download VADER if not already present
nltk.download("vader_lexicon", quiet=True)

# Initialize analyzers
vader = SentimentIntensityAnalyzer()
transformer = pipeline("sentiment-analysis")

# ---------------------- Helper Functions ----------------------
def fetch_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return " ".join(paragraphs)
    except Exception as e:
        return f"Error fetching URL: {e}"

def analyze_vader(text):
    scores = vader.polarity_scores(text)
    # Normalize to percentage
    total = scores["pos"] + scores["neu"] + scores["neg"]
    percentages = {
        "Positive": round((scores["pos"] / total) * 100, 2),
        "Neutral": round((scores["neu"] / total) * 100, 2),
        "Negative": round((scores["neg"] / total) * 100, 2),
    }
    overall = max(percentages, key=percentages.get)
    return overall, percentages

def analyze_transformer(text):
    result = transformer(text[:512])[0]  # first 512 chars for speed
    label = result["label"]
    score = round(result["score"] * 100, 2)
    percentages = {"Positive": 0, "Negative": 0, "Neutral": 0}
    if label.upper() == "POSITIVE":
        percentages["Positive"] = score
    elif label.upper() == "NEGATIVE":
        percentages["Negative"] = score
    else:
        percentages["Neutral"] = score
    return label.capitalize(), percentages

def plot_sentiment(percentages):
    fig, ax = plt.subplots()
    ax.bar(percentages.keys(), percentages.values(), color=["green", "blue", "red"])
    ax.set_title("Sentiment Analysis Result")
    ax.set_ylabel("Percentage")
    st.pyplot(fig)

# ---------------------- Streamlit UI ----------------------
st.title("📊 Sentiment Analysis Tool")
st.write("Enter text, URL, or upload a file to analyze sentiment.")

# Input options
input_type = st.radio("Choose input type:", ["Raw Text", "URL", "File Upload"])
backend = st.radio("Choose Analysis Method:", ["VADER (Fast)", "Transformer (Accurate)"])

text_data = ""

if input_type == "Raw Text":
    text_data = st.text_area("Enter your text here:")
elif input_type == "URL":
    url = st.text_input("Enter webpage URL:")
    if url:
        text_data = fetch_text_from_url(url)
elif input_type == "File Upload":
    file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])
    if file is not None:
        if file.name.endswith(".txt"):
            text_data = file.read().decode("utf-8")
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            st.write("CSV Preview:", df.head())
            column = st.selectbox("Select column to analyze", df.columns)
            text_data = " ".join(df[column].astype(str))

if st.button("🔍 Analyze Sentiment"):
    if text_data.strip() == "":
        st.warning("Please enter or upload some text.")
    else:
        if backend == "VADER (Fast)":
            overall, percentages = analyze_vader(text_data)
        else:
            overall, percentages = analyze_transformer(text_data)

        st.subheader("✅ Analysis Result")
        st.write(f"**Overall Sentiment:** {overall}")
        st.write("**Percentage Breakdown:**")
        for k, v in percentages.items():
            st.write(f"- {k}: {v}%")

        plot_sentiment(percentages)
