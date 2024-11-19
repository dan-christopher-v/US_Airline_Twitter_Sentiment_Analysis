import streamlit as st
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# Load the saved models
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('sentiment_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define a function to predict sentiment
def predict_sentiment(user_text):
    vectorized_text = vectorizer.transform([user_text])
    predicted_sentiment = model.predict(vectorized_text)
    predicted_label = label_encoder.inverse_transform(predicted_sentiment)[0]
    return predicted_label

# Streamlit App Layout
st.set_page_config(page_title="Airline Sentiment Analysis", layout="wide")

# App Header
st.title("Twitter US Airline Sentiment Analysis")
st.markdown("## Predict the sentiment of tweets related to US airlines")

# Add an image at the top
st.image('sentiment_model_image.jpg', use_column_width=True, caption="US Airline Sentiment Analysis")

# Sidebar for user input and options
st.sidebar.header("User Input")
st.sidebar.markdown("Enter a tweet related to US Airlines:")

# User input
user_text = st.sidebar.text_area("Tweet:", placeholder="Enter tweet here")

# Display sentiment prediction
if st.sidebar.button("Predict Sentiment"):
    if user_text:
        predicted_sentiment = predict_sentiment(user_text)
        st.sidebar.success(f"Predicted Sentiment: **{predicted_sentiment}**")
    else:
        st.sidebar.error("Please enter a tweet.")

# Add a section with example tweets
st.markdown("### Example Tweets")
example_tweets = ["I love flying with United!", "The flight was delayed for 4 hours. Horrible service.", 
                  "Excellent in-flight experience with JetBlue!"]
for tweet in example_tweets:
    st.write(f"- {tweet} (Predicted: **{predict_sentiment(tweet)}**)")

# Add a plot related to sentiment distribution (if you have data)
st.markdown("### Sentiment Distribution")
sentiments = ['positive', 'negative', 'neutral']
sentiment_counts = [2365, 9170, 3100]  # Example counts

fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=sentiments, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99','#ff6666'])
ax.axis('equal')
st.pyplot(fig)

# Add an image of the data pipeline (if available)
st.markdown("### Data Pipeline")
st.image('Blank diagram.png', use_column_width=True, caption="Overview of the Data Processing Pipeline")

# Footer
st.markdown("---")

