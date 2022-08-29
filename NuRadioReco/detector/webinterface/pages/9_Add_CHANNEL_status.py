import streamlit as st
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.detector_mongo import Detector

det = Detector(database_connection='test')

page_configuration()

old_name = st.text_input('input old name:')
st.markdown(old_name)
new_name = st.text_input('input new name:')
st.markdown(new_name)
rename = st.button('Rename')
if rename:
    det.rename_database_collection(old_name, new_name)
