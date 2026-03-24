import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# -------------------------------
# Load Model (SAFE WAY)
# -------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "simple_rnn_imdb.keras")

model = load_model(MODEL_PATH, compile=False)

# -------------------------------
# Load IMDB word index
# -------------------------------
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# -------------------------------
# Preprocess input text
# -------------------------------
def preprocess_text(text):
    words = text.lower().split()
    encoded = []

    for word in words:
        if word in word_index:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)  # unknown word

    padded = pad_sequences([encoded], maxlen=500)
    return padded

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎬 IMDB Sentiment Analysis (RNN)")
st.write("Enter a movie review below:")

user_input = st.text_area("✍️ Your Review")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        processed_input = preprocess_text(user_input)
        prediction = model.predict(processed_input)

        score = prediction[0][0]

        if score > 0.5:
            st.success(f"😊 Positive Review (Confidence: {score:.2f})")
        else:
            st.error(f"😞 Negative Review (Confidence: {score:.2f})")