import re
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
model = load_model("RNN/simple_rnn_imdb.h5")


def preprocess_text(review):
    review = review.lower()
    review = re.sub(r"[^a-zA-Z\s]", "", review)
    words = review.split()

    encoded_review = []
    for word in words:
        index = word_index.get(word, 2)
        encoded_review.append(index + 3)

    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, prediction


st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as Positive or Negative")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    if user_input.strip():
        sentiment, score = predict_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** {score:.4f}")
    else:
        st.warning("Please enter a movie review.")