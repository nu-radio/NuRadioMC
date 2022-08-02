import streamlit as st
from PIL import Image
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration

logo = Image.open('RNO-G_logo.png')


page_configuration()

st.title('Welcome to the RNO-G Hardware Database Uploader')
link = '[GitHub](https://github.com/RNO-G)'
link_wiki = '[here](https://radio.uchicago.edu/wiki/index.php/Main_Page)'
st.markdown(f'Here you can find the {link} page', unsafe_allow_html=True)
st.markdown(f'Or click {link_wiki} to get to the RNO-G wiki.')
st.image(logo)