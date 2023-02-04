import streamlit as st
import pandas as pd
import gdown
import pickle

###

def add_menu_title():
    """
    
    """
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "Application Menù";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
def data_retrieval(url):
    """
    
    """
    # download from gdrive
    file_id=url.split('/')[-2]
    dwn_url='https://drive.google.com/uc?id=' + file_id
    df = pd.read_pickle(dwn_url)
    return df
            