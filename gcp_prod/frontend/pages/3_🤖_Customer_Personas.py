import streamlit as st

from PIL import Image
from modules import utils
from streamlit_option_menu import option_menu
from modules import personas_single, personas_bulk, personas_retrain


###

app_logo = Image.open("images/logo.png")
icon = Image.open("images/icon.ico")

st.set_page_config(page_title="DIA - Customer Personas", layout="wide", page_icon=icon)

utils.add_menu_title()

st.image(app_logo)
st.markdown("# Customer Personas")

pages = option_menu(None, ["Single Analysis", "Bulk Analysis", "Retrain Model"],icons=['rocket-takeoff','database','recycle'], menu_icon="cast", default_index=0, orientation="horizontal")

if pages == "Single Analysis":
    personas_single.main()
elif pages == "Bulk Analysis":
    personas_bulk.main()
else:
    personas_retrain.main()