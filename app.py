import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download vader if not available
nltk.download("vader_lexicon", quiet=True)

# Transformers (optional)
try:
    from transformers import pipeline
    transformers_available = True
except:
    transformers_available = False


# ---------------- Utility functions ----------------
def fetch_text_from_url(url: str) -> str:
    """Fetch text content from a webpage."""
    resp = requests.get(url, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")
    article = soup.find("article")
    if article:
        text = " ".join([p.get_text() for p in article.find_all("p")])
    else:
        text = " ".join([p.get_text() for p in soup.find_all("p")])
    return re.sub(r"\s+", " ", text).strip()


def analyze_vader(text: str):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    return label, scores


def analyze_transformer(text: str):
    if not transformers_available:
        return "Transformers not installed", {}
    classifier = pipeline("sentiment-analysis")
    result = classifier(text[:512])[0]  # limit text length for demo
    return result["label"], {"score": result["score"]}


def analyze_text(text, model):
    if model == "VADER":
        return analyze_vader(text)
    else:
        return analyze_transformer(text)


# ---------------- Streamlit UI ----------------
st.title("📊 Sentiment Analysis App")

option = st.radio(
    "Select Input Source:",
    ["Enter Raw Text", "Provide URL", "Upload File", "Upload CSV"],
)

model = st.selectbox("Choose Model:", ["VADER", "Transformer (HuggingFace)"])


# Raw Text
if option == "Enter Raw Text":
    user_input = st.text_area("Enter your text here:")
    if st.button("Analyze Text"):
        if user_input.strip():
            label, scores = analyze_text(user_input, model)
            st.subheader("Result:")
            st.write(f"**Sentiment:** {label}")
            st.json(scores)
        else:
            st.warning("Please enter some text.")


# URL
elif option == "Provide URL":
    url = st.text_input("Enter webpage URL:")
    if st.button("Analyze URL"):
        if url:
            text = fetch_text_from_url(url)
            label, scores = analyze_text(text, model)
            st.subheader("Result:")
            st.write(f"**Sentiment:** {label}")
            st.json(scores)
            st.subheader("Extracted Text Preview:")
            st.write(text[:500] + ("..." if len(text) > 500 else ""))
        else:
            st.warning("Please enter a URL.")


# File Upload (TXT)
elif option == "Upload File":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file and st.button("Analyze File"):
        text = uploaded_file.read().decode("utf-8")
        label, scores = analyze_text(text, model)
        st.subheader("Result:")
        st.write(f"**Sentiment:** {label}")
        st.json(scores)
        st.subheader("File Preview:")
        st.write(text[:500] + ("..." if len(text) > 500 else ""))


# CSV Upload
elif option == "Upload CSV":
    uploaded_csv = st.file_uploader("Upload a CSV file", type=["csv"])
    text_column = st.text_input("Enter the column name containing text:")
    if uploaded_csv and text_column and st.button("Analyze CSV"):
        df = pd.read_csv(uploaded_csv)
        if text_column not in df.columns:
            st.error(f"Column '{text_column}' not found. Available: {list(df.columns)}")
        else:
            results = []
            for txt in df[text_column].fillna(""):
                label, scores = analyze_text(str(txt), model)
                results.append({"text": txt, "label": label, "score": scores})
            result_df = pd.DataFrame(results)
            st.subheader("CSV Results:")
            st.dataframe(result_df.head())
            st.download_button(
                "Download Results CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                "sentiment_results.csv",
                "text/csv",
            )
