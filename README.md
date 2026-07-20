# 🩺 Clinical Trial Disease Category Classification

An NLP and Machine Learning project that classifies clinical trial "Brief Summary" text into disease categories, deployed as an interactive Streamlit web app.

## 📌 Problem Statement

Clinical trial data contains large volumes of unstructured medical text. This project builds an end-to-end pipeline to automatically classify clinical trial summaries into disease categories using NLP and Machine Learning, supporting healthcare analytics and medical research organization.

## 🎯 Objective

- Clean and preprocess clinical trial text data
- Apply NLP techniques (tokenization, lemmatization, stop word removal)
- Convert text into numerical features using TF-IDF
- Train and compare multiple classification algorithms
- Deploy an interactive Streamlit app for real-time disease prediction

## 🗂️ Dataset

- **60,337 clinical trial records**
- Key columns: `brief_summary` (input text), `source_condition_query` (disease label)
- **8 disease categories:** Breast Cancer, COVID-19, Chronic Obstructive Pulmonary Disease, Rheumatoid Arthritis, Type 2 Diabetes, Glaucoma, Anxiety, Sickle Cell Anemia

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NLTK, Scikit-learn, Matplotlib, Seaborn, WordCloud, Joblib, Streamlit

## 🔄 Project Workflow

1. **Data Preprocessing** — text cleaning, stop word removal, lemmatization
2. **Exploratory Data Analysis** — category distribution, word frequency, word clouds
3. **Feature Engineering** — TF-IDF vectorization (top 5,000 features)
4. **Model Training & Comparison**

   | Model | Accuracy |
   |---|---|
   | **Logistic Regression** | **94.47%** ✅ (best) |
   | Random Forest | 93.97% |
   | Decision Tree | 91.05% |

5. **Model Deployment** — Streamlit app for live disease category prediction with confidence scores

## 📊 Results

- Best model: **Logistic Regression** — highest accuracy and fastest training time
- Balanced precision/recall across all 8 disease categories
- Deployed as an interactive web app with real-time prediction and confidence visualization

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/clinical-trial-disease-classification.git
cd clinical-trial-disease-classification

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 📁 Project Structure
