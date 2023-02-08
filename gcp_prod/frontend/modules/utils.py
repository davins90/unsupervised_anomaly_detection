import streamlit as st
import pandas as pd
import gdown
import pickle
import plotly.graph_objects as go

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
    
@st.cache
def data_retrieval(url):
    """
    
    """
    # download from gdrive
    file_id=url.split('/')[-2]
    dwn_url='https://drive.google.com/uc?id=' + file_id
    df = pd.read_pickle(dwn_url)
    return df
            
def gauge_chart(val):
    """
    
    """
    fig = go.Figure(go.Indicator(mode = "gauge+number",value = val, 
                                     domain = {'x': [0, 1], 'y': [0, 1]},
                                     gauge = {'axis': {'range': [0.0, 1.0]},'bar': {'color': "darkblue"},
                                             'steps': [
            {'range': [0.0, 0.5], 'color': 'green'},
            {'range': [0.5, 0.8], 'color': 'orange'},
            {'range': [0.8, 1.0], 'color': 'red'}]},
                                     title = {'text': "Final Fraud Score"}))
    fig.show()
    return st.plotly_chart(fig)