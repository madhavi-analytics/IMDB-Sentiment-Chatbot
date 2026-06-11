# IMDB-Sentiment-Chatbot

## 📌 Project Overview
MovieMind AI is an NLP-powered chatbot that analyzes IMDb movie reviews and predicts whether the sentiment is **Positive 😊** or **Negative 😞** in real time.

The chatbot uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to understand movie review text and provide sentiment predictions with confidence scores.

---

## 🚀 Features
✅ Chatbot-style interface  
✅ Real-time sentiment prediction  
✅ Positive & Negative classification  
✅ Confidence score prediction  
✅ NLP preprocessing pipeline  
✅ Machine Learning-based sentiment analysis  
✅ User-friendly interactive UI  

---

## 🎯 Problem Statement
Reading thousands of movie reviews manually is time-consuming. This chatbot helps automatically analyze movie reviews and identify audience sentiment instantly.

---

## 📂 Dataset
- **Dataset:** IMDb Movie Reviews Dataset  
- **Total Reviews:** 13,559  
- **Sentiment Classes:** Positive & Negative

---

## 🛠 Technologies Used
- Python
- NLP (Natural Language Processing)
- NLTK
- Scikit-learn
- Streamlit
- Joblib
- Pandas
- NumPy

---

## 🔍 NLP Preprocessing Steps
The following preprocessing techniques were applied:

- Lowercase conversion
- HTML tag removal
- Remove numbers
- Remove special characters
- Remove extra whitespaces
- Tokenization
- Stopwords removal
- Lemmatization

---

## ⚙ Feature Engineering
### TF-IDF Vectorization
TF-IDF was used to convert text reviews into numerical vectors for machine learning prediction.

---

## 🤖 Machine Learning Model
**Naive Bayes Classifier**

Why Naive Bayes?
- Fast and efficient for text classification
- Works well with NLP datasets
- High performance on sparse text data

**Model Accuracy:** ~86%

---

## 💬 Chatbot Workflow
1. User enters a movie review  
2. Chatbot preprocesses the text  
3. TF-IDF converts text into numerical format  
4. Machine learning model predicts sentiment  
5. Chatbot displays result with confidence score  

---

## 📸 Example

**User:**  
> This movie was amazing and fantastic

**Bot:**  
> 😊 Positive Review  
> Confidence: 90%

**User:**  
> Waste movie, boring story

**Bot:**  
> 😞 Negative Review  
> Confidence: 62%

---

## 💡 Business Use Case
Movie production companies and streaming platforms can use sentiment analysis to quickly understand audience feedback without manually reviewing thousands of comments.

---

## 🔮 Future Improvements
- Add multilingual review support  
- Improve chatbot intelligence  
- Add review summary feature  
- Deploy advanced AI conversational chatbot
