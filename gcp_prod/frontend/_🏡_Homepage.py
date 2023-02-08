import streamlit as st

from PIL import Image
from modules import utils

###

app_logo = Image.open("images/logo.png")
icon = Image.open("images/icon.ico")

st.set_page_config(page_title="DIA - Homepage", layout="wide", page_icon=icon)

utils.add_menu_title()

st.image(app_logo)
st.markdown("# Decision Intelligence Application")
st.markdown("## Introduction")

st.markdown("This application was created with the aim of providing an analysis tool to meet two specific objectives: \n - 1) Given a certain transaction, what is the probability that it is fraudulent? \n - 2) Given a certain customer, what type of customer is it? \n As can be seen from the previous business objectives, the users of this tool can be of two types: \n - 1) the fraud analysis team, to detect a fraudulent transaction early, either by analyzing a single transaction or by analyzing a set of transactions; \n - 2) the marketing team, to analyze customers and better tailor services to them. \n Through browsing the pages of the POC now designed, it is possible to: \n - 1) perform exploratory data analysis, to extract descriptive characteristics of the user base in its current state. \n - 2) Obtain a score, representing the fraud probability of a transaction, based on the input data provided. \n - 3) Obtain the marketing class to which a customer belongs to understand how best to personalize the service. \n - 4) Read the available documentation of the project.")



