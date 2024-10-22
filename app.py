import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
word_index = imdb.get_word_index()
model = load_model('SIMPLERNN2E.h5')
def preprocess(text):
    text =text.lower()
    print(text)
    ar = text.split(" ")
    my   = [word_index.get(word , -1) + 3  for word in ar]
    s1 = sequence.pad_sequences([my] , maxlen = 500)
    #print(s1)
    prediction = model.predict(s1)
    print(prediction)
    if prediction[0][0] > 0.6 :
        st.write("Good Review")
    else:
        st.write("Bad Review")
st.write("Review Anaylysis")
with st.form("my_form"):
    text = st.text_input("Enter the Review")
    submitted = st.form_submit_button("Submit")
    if submitted:
        preprocess(text)