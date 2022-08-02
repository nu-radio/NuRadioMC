import streamlit as st
from PIL import Image


def page_configuration():
    logo = Image.open('RNO-G_logo.png')
    menu_dic = {'Get Help': 'https://radio.uchicago.edu/wiki/index.php/Main_Page', 'Report a bug': "https://github.com/RNO-G", 'About': "Cool RNO-G calibration datatbase"}
    st.set_page_config(layout="wide", page_icon=logo, page_title='RNO-G Hardware Database', menu_items=menu_dic)
    st.empty()
    st.sidebar.image(logo, use_column_width=True)