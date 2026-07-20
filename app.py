
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------- Load saved model, vectorizer, and label encoder ----------------
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

# ---------------- Text cleaning function (must match training) ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Clinical Trial Disease Classifier",
    page_icon="🩺",
    layout="centered"
)

# ---------------- Dark theme styling ----------------
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stTextArea textarea {
        background-color: #1c1f26;
        color: #fafafa;
    }
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.title("🩺 Clinical Trial Disease Category Classifier")
st.write("Paste a clinical trial **brief summary** below, and the model will predict which of the 8 disease categories it belongs to.")

with st.expander("ℹ️ About this app"):
    st.write("""
    This app uses a **Logistic Regression** model trained on 60,000+ clinical trial summaries.
    Text is cleaned (lowercased, stopwords removed, lemmatized) and converted into numeric
    features using **TF-IDF**, then classified into one of 8 disease categories:
    Anxiety, Breast Cancer, COPD, COVID-19, Glaucoma, Rheumatoid Arthritis, Sickle Cell Anemia, Type 2 Diabetes.
    """)

# ---------------- Input ----------------
user_input = st.text_area("Enter Clinical Trial Brief Summary:", height=200, placeholder="e.g. This study evaluates the effect of a new inhaler on patients with chronic breathing difficulty...")

# ---------------- Predict button ----------------
if st.button("🔍 Predict Disease Category"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])

        pred_num = model.predict(vec)[0]
        pred_label = le.inverse_transform([pred_num])[0]
        probs = model.predict_proba(vec)[0]

        st.success(f"### Predicted Disease Category: **{pred_label.title()}**")

        prob_df = pd.DataFrame({
            "Disease": le.classes_,
            "Confidence": probs
        }).sort_values("Confidence", ascending=False)

        st.subheader("Confidence across all categories")
        st.bar_chart(prob_df.set_index("Disease"))

        with st.expander("🧹 See cleaned text (what the model actually reads)"):
            st.code(cleaned)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Clinical Trial Disease Category Classification | NLP + Machine Learning Project")