import os
import re
import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "simple_rnn_imdb.keras")
model = load_model(MODEL_PATH, compile=False)
word_index = imdb.get_word_index()
MAXLEN = 500
def preprocess_text(review: str):
    review = review.lower()
    review = re.sub(r"[^a-zA-Z\s]", "", review)
    words = review.split()

    encoded_review = []
    for word in words:
        index = word_index.get(word, 2)   # 2 = unknown word
        encoded_review.append(index + 3)  # IMDB offset

    padded_review = pad_sequences([encoded_review], maxlen=MAXLEN)
    return padded_review

# -------------------------------
# Predict sentiment
# -------------------------------
def predict_sentiment(review: str):
    processed_input = preprocess_text(review)
    prediction = model.predict(processed_input, verbose=0)[0][0]

    if prediction >= 0.5:
        sentiment = "Positive Review 😊"
        confidence = prediction
    else:
        sentiment = "Negative Review 😟"
        confidence = 1 - prediction

    return sentiment, prediction, confidence

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎬 IMDB Sentiment Analysis (RNN)")
st.write("Enter a movie review below:")

user_input = st.text_area("✍️ Your Review")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        sentiment, raw_score, confidence = predict_sentiment(user_input)

        st.write(f"**Raw Model Score:** {raw_score:.4f}")

        if raw_score >= 0.5:
            st.success(f"{sentiment} (Confidence: {confidence:.2f})")
        else:
            st.error(f"{sentiment} (Confidence: {confidence:.2f})")