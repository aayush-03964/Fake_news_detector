# 📰 AI-Driven Fake News Detector

A production-ready, end-to-end Natural Language Processing (NLP) web application that utilizes a machine learning classifier to assess the statistical probability of a news headline being authentic or unverified. 

### 🚀 [Live App Deployment Link](https://fakenewsdetector-a5cwm6wrwg6k9n33srtsqp.streamlit.app/)

---

## 🛠️ System Architecture & Engineering Pipeline

The application processes raw text input through a serialized machine learning pipeline that handles sparse matrix representation and real-time inference.

1. **Text Preprocessing:** Standardizes raw input by shifting text to lowercase, stripping out URLs, removing punctuation/special characters via Regex, and filtering out high-frequency English stopwords.
2. **Feature Extraction:** Transforms tokens into a dense numeric feature space using a **TF-IDF Vectorizer** capped at 5,000 maximum features to prevent overfitting and limit dimensionality.
3. **Classification Engine:** Evaluates the vector distributions using a calibrated **Logistic Regression** model to predict binary authenticity (`1` for Authentic, `0` for Unverified) along with exact class probabilities.
4. **Cloud Interface:** Deployed via **Streamlit Cloud** to serve the serialized pickle artifacts (`model.pkl` and `vectorizer.pkl`) with cached system resource allocation.

---

## 📊 Performance & Validation Metrics

The underlying machine learning architecture handles data variance smoothly across distinct linguistic distributions:
* **Sensationalized Clickbait / Viral Rumors:** Successfully flagged as *Unverified / Fake News* with confidence boundaries reaching **88% - 89%**.
* **Standard Journalistic Wire Reports:** Correctly classified as *Authentic News* with exceptional high-fidelity confidence scoring up to **97.78%**.

---

## 💻 Tech Stack & Libraries Used

* **Language:** Python
* **Machine Learning & Vectorization:** Scikit-Learn (`pickle`, `LogisticRegression`, `TfidfVectorizer`)
* **Natural Language Processing:** NLTK (`stopwords`)
* **Regular Expressions:** `re`
* **Web Deployment Framework:** Streamlit
