import streamlit as st
import pickle

with open("svm_pipeline.pkl", "rb") as f:
    svm_pipeline = pickle.load(f)

label_map = {0: "Negative", 1: "Positive", 2: "Neutral/Mixed"}

st.set_page_config(page_title="Sentiment Classifier")
st.title("💬 Cimas Sentiment Classification")
st.write("Enter a comment and classify it as Positive, Negative, or Neutral.")

comment = st.text_area("Your Comment/Isai Comment Yenyu:")

if st.button("Classify"):
    if comment.strip():
        prediction = svm_pipeline.predict([comment])[0]
        st.success(f"Predicted Sentiment: **{label_map[prediction]}**")
    else:
        st.warning("Please enter a valid comment.")
