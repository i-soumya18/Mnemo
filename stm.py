import streamlit as st

# Streamlit UI
st.title("👋 Welcome to Streamlit App")

# Text input
name = st.text_input("Enter your name:")

if st.button("Submit"):
    if name.strip():
        st.success(f"Hello, {name}! Welcome to the Streamlit app.")
    else:
        st.warning("Please enter your name.")