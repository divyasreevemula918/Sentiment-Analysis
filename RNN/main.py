# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import re
# word_index=imdb.get_word_index()
# reverse_word_index={value: key for key,value in word_index.items()}
# model=load_model('simple_rnn_imdb.keras')
# def decode_review(encoded_review):
#     return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])
# def preprocess_text(review):
#     review = review.lower()
#     review = re.sub(r"[^a-zA-Z\s]", "", review)
#     words = review.split()

#     encoded_review = []
#     for word in words:
#         index = word_index.get(word, 2)   # unknown word
#         encoded_review.append(index + 3)  # IMDB offset

#     padded_review = pad_sequences([encoded_review], maxlen=500)
#     return padded_review
# def predict_sentiment(review):
#     preprocessed_input = preprocess_text(review)
#     prediction = model.predict(preprocessed_input)[0][0]

#     sentiment = "Positive" if prediction > 0.5 else "Negative"
#     return sentiment, prediction
# import streamlit as st
# st.title("IMDB Movie Review Sentiment Analysis")
# st.write("Enter a movie review to classify it as positive or negative")
# user_input=st.text_area("Movie Review")
# if st.button("clasify"):
#     preprocessed_input=preprocess_text(user_input)
#     prediction=model.predict(preprocessed_input)
#     sentiment="positive" if prediction[0][0]>0.5 else "Negative"
#     st.write(f"Sentiment:{sentiment}")
#     st.write(f"Prediction score:{prediction[0][0]}")
# else:
#     st.write("please enter a movie review.")
import re
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load word index and model
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
model = load_model("simple_rnn_imdb.keras")


def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])


def preprocess_text(review):
    review = review.lower()
    review = re.sub(r"[^a-zA-Z\s]", "", review)
    words = review.split()

    encoded_review = []
    for word in words:
        index = word_index.get(word, 2)   # unknown word
        encoded_review.append(index + 3)  # IMDB offset

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
    if user_input.strip() != "":
        sentiment, score = predict_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** {score:.4f}")
    else:
        st.warning("Please enter a movie review.")