import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("imdb_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page title
st.title("🎬 MovieMind AI Chatbot")

st.write("Talk to the movie review bot!")

# User input
user_review = st.chat_input("Type your movie review...")

if user_review:

    # Show user message
    with st.chat_message("user"):
        st.write(user_review)

    # Prediction
    review_vector = tfidf.transform([user_review])
    prediction = model.predict(review_vector)
    confidence = model.predict_proba(review_vector)

    # Bot response
    with st.chat_message("assistant"):

        if prediction[0] == 1:
    conf = min(confidence[0][1] * 100, 90)

    st.success(
        f"😊 Positive Review\n\nConfidence: {conf:.0f}%"
    )

else:
    conf = max(50, min(confidence[0][0] * 100, 70))

    st.error(
        f"😞 Negative Review\n\nConfidence: {conf:.0f}%"
    )
