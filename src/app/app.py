import streamlit as st
import requests

st.title("Sum Calculator")

a = st.number_input("Enter a number:")
b = st.number_input("Enter another number:")

a = int(a)
b = int(b)

if st.button("Submit"):
    payload = {"a": a, "b": b}
    result = requests.post(f"http://fast_api:8000/sum/",json=payload).json()
    st.write(result)