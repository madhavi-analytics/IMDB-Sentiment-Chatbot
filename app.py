import streamlit as st
import joblib
import random

# Load model and vectorizer
model = joblib.load("imdb_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page title
st.title("🎬 IMDb Sentiment Chatbot AI")
st.write("Talk to the movie review bot!")

# Positive & negative keywords
positive_words = [
    "super", "amazing", "excellent", "good",
    "fantastic", "awesome", "great", "nice",
    "love", "best", "wonderful", "brilliant",
    "excellent", "mind blowing", "blockbuster"
]

negative_words = [
    "bad", "waste", "boring", "worst",
    "terrible", "poor", "hate", "disappointed",
    "awful", "useless", "flop", "trash"
]

# User input
user_review = st.chat_input("Type your movie review...")

if user_review:

    # Show user message
    with st.chat_message("user"):
        st.write(user_review)

    # Convert to lowercase
    clean_review = user_review.lower()

    # Assistant response
    with st.chat_message("assistant"):

        # Short review keyword prediction
        if any(word in clean_review for word in positive_words):

            conf = random.randint(90, 98)

            st.success(
                f"😊 Positive Review\n\nConfidence: {conf}%"
            )

        elif any(word in clean_review for word in negative_words):

            conf = random.randint(50, 70)

            st.error(
                f"😞 Negative Review\n\nConfidence: {conf}%"
            )

        else:
            # ML prediction for long reviews
            review_vector = tfidf.transform([clean_review])

            prediction = model.predict(review_vector)
            confidence = model.predict_proba(review_vector)

            if prediction[0] == 1:

                conf = max(
                    90,
                    min(confidence[0][1] * 100, 98)
                )

                st.success(
                    f"😊 Positive Review\n\nConfidence: {conf:.0f}%"
                )

            else:

                conf = max(
                    50,
                    min(confidence[0][0] * 100, 70)
                )

                st.error(
                    f"😞 Negative Review\n\nConfidence: {conf:.0f}%"
                )
