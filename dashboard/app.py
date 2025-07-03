# -*- coding: utf-8 -*-
import streamlit as st
import requests
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="DBPedia Classifier",
    page_icon="🤖",
    layout="wide"
)

# --- Helper Functions for EDA ---
def generate_wordcloud(text):
    """Generates and displays a WordCloud from the input text."""
    # Simple text cleaning: remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text).lower()
    
    # Generate WordCloud object
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis', # This colormap is perceptually uniform and good for colorblindness
        stopwords=None,
        collocations=False
    ).generate(text)

    # Display the generated image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def get_word_frequency(text):
    """Calculates word frequency and returns a DataFrame."""
    words = re.sub(r'[^\w\s]', '', text).lower().split()
    freq_df = pd.DataFrame(words, columns=['word'])
    freq_df = freq_df['word'].value_counts().reset_index()
    freq_df.columns = ['Word', 'Frequency']
    return freq_df.head(15)


# --- Application Title and Description ---
st.title("📄 Text Classifier for DBPedia")
st.markdown("""
This dashboard demonstrates a model for classifying text into DBPedia's Level 1 categories. 
Enter a piece of text below, and the model will predict its primary category and perform a basic textual analysis.
""")

# --- API Configuration ---
API_URL = "https://api-dbpedia-poc.wonderfulpebble-3f07ef82.francecentral.azurecontainerapps.io/predict"

# --- User Input Section ---
st.header("Classify Your Text")
input_text = st.text_area(
    label="Enter text for classification and analysis:", # WCAG: Clear and descriptive label
    height=150,
    placeholder="Example: The Eiffel Tower is a wrought-iron lattice tower...",
    help="Paste any English text here. For a full analysis including a WordCloud, use at least 50 words." # WCAG: Provides extra guidance
)

# --- Prediction & EDA Logic ---
if st.button("Analyze and Classify", type="primary"):
    if input_text and len(input_text.strip()) > 10:
        
        # --- 1. API Call for Prediction ---
        with st.spinner("Classifying text..."):
            try:
                payload = {"text": input_text}
                response = requests.post(API_URL, json=payload, timeout=30)
                response.raise_for_status()
                prediction_data = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the classification API: {e}")
                st.info(f"Please ensure the FastAPI server is running at `{API_URL}`.")
                prediction_data = None
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                prediction_data = None

        # --- 2. Display Prediction Results ---
        if prediction_data:
            predicted_class = prediction_data.get("predicted_class", "N/A")
            probabilities = prediction_data.get("probabilities", {})

            st.subheader("✅ Prediction Result")
            st.success(f"**Predicted Category:** `{predicted_class}`")

            st.subheader("📊 Class Probabilities")
            prob_df = pd.DataFrame(
                list(probabilities.items()),
                columns=['Category', 'Probability']
            ).sort_values(by='Probability', ascending=False).reset_index(drop=True)
            
            st.bar_chart(prob_df.set_index('Category'))
        
        st.divider()

        # --- 3. On-the-fly Exploratory Data Analysis (EDA) ---
        st.subheader("📝 Exploratory Analysis of Your Text")
        
        word_count = len(input_text.split())
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Word Count", value=word_count)
            st.write("**Top 15 Most Frequent Words**") # WCAG: Clear heading for the chart
            freq_df = get_word_frequency(input_text)
            st.bar_chart(freq_df.set_index('Word'))
            
        with col2:
            st.write("**Word Cloud**") # WCAG: Clear heading for the visual
            if word_count > 50: 
                with st.spinner("Generating WordCloud..."):
                    generate_wordcloud(input_text)
                    # WCAG: Caption provides a text alternative for the image
                    st.caption("A WordCloud visualizing the most frequent words in the provided text. Larger words appear more often.")
            else:
                st.info("Enter more than 50 words to generate a WordCloud.")
    else:
        st.warning("Please enter some text to classify (at least 10 characters).")