import streamlit as st
import joblib
import re
from PIL import Image

# Load the trained model and TF-IDF vectorizer
model = joblib.load('best_genre_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Set page configuration
st.set_page_config(
    page_title="Movie Genre Predictor",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom background image
def add_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://images.unsplash.com/photo-1510275770273-e0bc6559a5e3");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_image()

# App title and description
st.title("🎬 Movie Genre Predictor 🍿")
st.subheader("Enter your movie description and let the magic happen!")

# User input
user_input = st.text_area("📝 Movie Description", "", placeholder="Type a movie description here...")

# Fun elements
if st.button("🎥 Predict Genre"):
    if user_input:
        with st.spinner("Analyzing your movie description... 🍿"):
            processed_description = preprocess_text(user_input)
            vectorized_description = tfidf.transform([processed_description])
            predicted_genre = model.predict(vectorized_description)
            st.success(f"**Predicted Genre:** {predicted_genre[0]}")
            st.balloons()
    else:
        st.warning("Please enter a movie description to predict its genre.")


st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>Created by <a href="https://github.com/Anikesh20" target="_blank">Anikesh Kumar Singh</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

