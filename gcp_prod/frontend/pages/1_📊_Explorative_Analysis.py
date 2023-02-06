import streamlit as st
import pandas as pd


from PIL import Image
from modules import utils, ex_local_upload, ex_connect
from streamlit_option_menu import option_menu

###

app_logo = Image.open("images/logo.png")
icon = Image.open("images/icon.ico")

st.set_page_config(page_title="DIA - Explorative Analysis", layout="wide", page_icon=icon)

utils.add_menu_title()

st.image(app_logo)
st.markdown("# Explorative Analysis")

pages = option_menu(None, ["Connect to DB","Local Upload"],icons=['cloud-download','upload'], menu_icon="cast", default_index=0, orientation="horizontal")

if pages == "Local Upload":
    ex_local_upload.main()
else:
    ex_connect.main()
    