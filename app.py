import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# For Stemming
ps = PorterStemmer()

# Load model and vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file, open('model.pkl', 'rb') as model_file:
    tfidf = pickle.load(vectorizer_file)
    model = pickle.load(model_file)


def transform_msg(msg):
    msg = msg.lower()  # Lower Case
    msg = nltk.word_tokenize(msg)  # Tokenization

    # Appending all the words except special characters.
    y = [i for i in msg if i.isalnum()]

    msg = y[:]
    y.clear()

    # Appending all the words except stopwords & punctuations.
    for i in msg:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    msg = y[:]
    y.clear()

    # Stemming
    for i in msg:
        y.append(ps.stem(i))

    return " ".join(y)


# Customizing Streamlit app layout and settings
st.set_page_config(
    page_title="Spam Classifier App",
    page_icon=":email:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title and sidebar
st.title("Spam or Not?")
st.sidebar.title("About")
st.sidebar.info(
    "This spam classification app is built by Divyang Jha, a final year Engineering student with strong communicational & analytical skills, proficient in Python & Machine Learning. Connect with Divyang:\n\n"
    "[LinkedIn Profile](https://www.linkedin.com/in/divyang-jha-6b3451201/)\n\n"
    "[LeetCode Profile](https://leetcode.com/divyang_2222/)\n\n"
    "[GeeksforGeeks Profile](https://auth.geeksforgeeks.org/user/divyang_2222)\n\n"
    "Having experience using libraries like pandas, NumPy, Scikit-learn, etc., familiar with AI, NLP & Front-end technologies like HTML & CSS."
)

# Input area
input_sms = st.text_area("Enter your message here")

if st.button('Predict'):
    if input_sms:
        # Preprocess the input message
        transformed_sms = transform_msg(input_sms)
        # Vectorize the input
        vector_input = tfidf.transform([transformed_sms])
        # Make prediction
        result = model.predict(vector_input)[0]
        # Display result
        if result == 1:
            st.error("This looks like a **Spam Message!** :warning:")
            st.info(
                "Some suggestions to identify spam messages:\n\n"
                "- Check the sender's email address for authenticity.\n"
                "- Verify links before clicking by hovering over them.\n"
                "- Examine the email content for grammar mistakes and urgency.\n"
                "- Avoid providing personal information or clicking on suspicious links.\n"
                "- Use security software to filter potential threats.\n"
                "- Trust your instincts; if it seems suspicious, double-check or ignore it."
            )
        else:
            st.success("This seems **Not Spam!** :white_check_mark:")
    else:
        st.warning("Please input a message to predict.")
