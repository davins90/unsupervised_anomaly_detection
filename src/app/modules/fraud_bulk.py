def main():
    import streamlit as st
    import gdown
    import pandas as pd
    
    
    st.title("fraud bulk")
    
    # download from gdrive
    url = "https://drive.google.com/file/d/1av_NaF0lQqhQOJewFA0NgRzEl9M9Qfgv/view?usp=share_link"
    file_id=url.split('/')[-2]
    dwn_url='https://drive.google.com/uc?id=' + file_id
    df = pd.read_pickle(dwn_url)
    
    st.write(df.head(1))