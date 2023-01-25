import streamlit as st

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
            