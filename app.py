import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords (for local first-time use only)
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)           # Remove non-letters
    text = text.lower().split()                     # Lowercase + split
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Enter a news **headline or paragraph**, and this app will predict whether it's **Real** or **Fake**.")

user_input = st.text_area("🔍 Enter News Text Here", height=200)

if st.button("🔎 Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Preprocess and predict
        clean_text = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([clean_text])
        print(vectorized_input)
        prediction = model.predict(vectorized_input)[0]

        if prediction == 0:
            st.success("✅ This news is **Real**.")
        else:
            st.error("🚨 This news is **Fake**.")
