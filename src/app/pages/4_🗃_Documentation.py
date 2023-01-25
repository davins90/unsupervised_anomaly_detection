import streamlit as st

from PIL import Image
from modules import utils

###

app_logo = Image.open("app/images/logo.png")
icon = Image.open("app/images/icon.ico")

st.set_page_config(page_title="DIA - Docs", layout="wide", page_icon=icon)

utils.add_menu_title()

st.image(app_logo)
st.markdown("# Documentation")