# 😁😶☹️Sentiment Analysis Tool

This Streamlit application is a versatile **Sentiment Analysis Tool** that allows users to analyze the emotional tone of text. You can input text directly, fetch content from a URL, or upload a file (`.txt` or `.csv`). The application provides two powerful analysis engines: **VADER** for fast, rule-based analysis and a **Transformer model** for more accurate, context-aware analysis.

-----

## Features

  - **Multiple Input Methods**: Analyze text from three sources:
      - **Raw Text**: Paste any text directly into the text area.
      - **URL**: Provide a URL to scrape text content from a webpage.
      - **File Upload**: Upload `.txt` or `.csv` files for analysis.
  - **Two Analysis Engines**: Choose the best tool for your needs:
      - **VADER**: A fast, lexicon-based model great for quick analysis of social media text.
      - **Transformer**: A more accurate, deep-learning model (based on `cardiffnlp/twitter-roberta-base-sentiment-latest`) that understands nuances and context.
  - **Detailed Results**: Get a breakdown of sentiment percentages (Positive, Neutral, Negative) and an overall sentiment classification.
  - **Interactive Visualization**: The results are presented in a clear, easy-to-read bar chart.

-----

## Getting Started

Follow these steps to set up and run the application on your local machine.

### Prerequisites

You'll need Python 3.8+ and `pip` installed.

### Installation

1.  **Clone the repository**:

    ```sh
    git clone https://github.com/mithun-27/sentiment_analysis.git
    ```

2.  **Create a virtual environment** (recommended):

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries**:

    ```sh
    pip install -r requirements.txt
    ```

    *Note: The `requirements.txt` file should contain all the libraries used in the code, which are: `streamlit`, `requests`, `beautifulsoup4`, `nltk`, `pandas`, `matplotlib`, `transformers`, `torch`.*

### Running the App

1.  **Run the Streamlit application**:

    ```sh
    streamlit app.py
    ```

    (Replace `your_script_name.py` with the name of the Python file containing the code.)

2.  The application will open in your default web browser.

-----

## How to Use

1.  Select an input tab: **Raw Text**, **URL**, or **File Upload**.
2.  Provide your text, URL, or file in the selected tab.
3.  Click the corresponding button to load the data (e.g., "Fetch from URL").
4.  Once the text is loaded, a new section will appear. Choose your analysis engine: **VADER** or **Transformer**.
5.  Click the "Analyze" button for your chosen model to view the results.

-----

## Screenshots

<img width="1919" height="874" alt="image" src="https://github.com/user-attachments/assets/e2015bc0-393a-4713-9d6c-c35cb96553cd" />

<img width="1919" height="866" alt="image" src="https://github.com/user-attachments/assets/74767dd1-1be2-4330-8836-52a7fd9ce023" />

<img width="1919" height="880" alt="image" src="https://github.com/user-attachments/assets/016893bd-2878-48aa-a771-6898921e2c82" />



-----

Note on URLs: This tool is designed to work with unsecured or less-secured URLs (e.g., those without complex authentication or strong anti-scraping measures). Highly secured websites may not be accessible.

## Acknowledgment

  - **VADER** (Valence Aware Dictionary and sEntiment Reasoner) from NLTK.
  - **Hugging Face Transformers** for the `cardiffnlp/twitter-roberta-base-sentiment-latest` model.
  - **Streamlit** for the web application framework.
