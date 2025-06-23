import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


# Download stopwords (only the first time)
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Load dataset
df = pd.read_csv("fake_train.csv")  # Make sure this file is in the same folder
df = df.fillna('')
df['content'] = df['author'] + ' ' + df['title']

# Preprocessing function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)           # Remove non-letters
    text = text.lower().split()                     # Lowercase + split
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

# Apply preprocessing
df['content'] = df['content'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['content'])  # remove .toarray()
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train SVM model
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(svm_model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ SVM model and vectorizer saved successfully.")
