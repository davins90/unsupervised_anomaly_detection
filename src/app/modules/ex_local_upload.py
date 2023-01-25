def main():
    import streamlit as st
    import pandas as pd
    
    st.title("local")
    file = st.file_uploader("Upload a file", type=["csv", "txt", "xlsx", "pkl"])

    if file is not None:
        st.write("File uploaded: ", file.name)
        data = pd.read_pickle(file)
        st.write(data.head())