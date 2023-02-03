import copy

import streamlit as st
import pandas as pd
from plotly import subplots
import plotly.graph_objs as go
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page
from NuRadioReco.detector.webinterface.utils.helper_contact import load_contact_information_from_db, insert_contact_information_into_db, check_inserted_config_information

page_name = 'contact information'


def build_main_page(main_cont):
    main_cont.title('Add your contact information')
    main_cont.markdown('On this page, you can insert your name and email address.')
    main_cont.markdown('These information are needed in order to have a person to contact if there are some problems with the measurement.')
    main_cont.markdown('You only have ot input your data once.')

    contact_name = main_cont.text_input('Please insert your name.')
    email_address = main_cont.text_input('Please insert your email address.')

    # validate inputs
    validated = check_inserted_config_information(main_cont, contact_name, email_address)

    # INSERT button
    upload_button = main_cont.button('INSERT TO DB', disabled=validated)

    # insert the data into the database and change to the success page by setting the session key
    if upload_button:
        insert_contact_information_into_db(name=contact_name, email=email_address)
        main_cont.empty()
        st.session_state.key = '1'
        st.experimental_rerun()


# main page setup
page_configuration()

# initialize the session key (will be used to display different pages depending on the button clicked)
if 'key' not in st.session_state:
    st.session_state['key'] = '0'

main_container = st.container()  # container for the main part of the page (with all the input filed)
success_container = st.container()  # container to display the page when the data was submitted

if st.session_state.key == '0':
    build_main_page(main_container)  # after clicking the submit button, the session key is set to '1' and the page is rerun

if st.session_state.key == '1':
    build_success_page(success_container, page_name)  # after clicking the 'add another measurement' button, the session key is set to '0' and the page is rerun
