import streamlit as st
import pickle

# Load the saved vectorizer and naive model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
classifiers = pickle.load(open('random_forest.pkl', 'rb'))

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')
ps = PorterStemmer()



def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize
    text = [word for word in text if word.isalnum()]  # Remove special characters and retain alphanumeric words
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]  # Remove stopwords and punctuations
    text = [ps.stem(word) for word in text]  # Apply stemming
    return " ".join(text)

# Set page title and icon
st.set_page_config(page_title="SMS Spam Detector", page_icon=":iphone:")

# Set app title and description
st.title("SMS Spam Detector")

st.markdown("""
            <p style="color: #1f2e2e;font-size:25px">This app predicts whether a given text message is spam or not.
            Please enter your text message and click the 'Predict' button.</p>
            """,
            unsafe_allow_html=True)

# Input area for the user's text message
input_sms = st.text_area("Enter your text here:")

# Predict button
if st.button('Detect'):
    # Check if input is empty
    if input_sms.strip() == '':
        st.error("Please enter a text message.")
    else:
        # Preprocess the text
        transformed_sms = transform_text(input_sms)
        # Vectorize the text
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = classifiers.predict(vector_input)[0]
        # Display prediction result with appropriate color
        if result == 1:
            st.error("SMS is Spam!!!")
        else:
            st.success("SMS is Not Spam.")

# Set footer with link to GitHub repository
st.markdown("""
            <hr>
            <p style="text-align:center; font-size:25px">Developed by Siwani Karna.</p>
            <p style="text-align:center">View on GitHub(https://github.com/Siwanikarna/SMS-Classifier)</p>
            """,
            unsafe_allow_html=True)
