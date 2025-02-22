import streamlit as st
from transformers import pipeline

# Load summarization pipeline
@st.cache_resource()
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Streamlit UI
st.title("📜 AI-Powered Text Summarizer")
st.write("Enter long-form text, and get a concise summary.")

# Text input
input_text = st.text_area("Enter your text here:", height=200)

if st.button("Summarize"):
    if input_text.strip():
        with st.spinner("Generating summary..."):
            summary = summarizer(input_text, max_length=150, min_length=50, do_sample=False)
            st.success("Summary:")
            st.write(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text before summarizing.")

# Footer
st.caption("Powered by Hugging Face Transformers & Streamlit")
