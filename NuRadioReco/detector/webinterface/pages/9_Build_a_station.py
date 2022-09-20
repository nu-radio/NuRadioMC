import copy

import streamlit as st
import pandas as pd
from plotly import subplots
import plotly.graph_objs as go
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page
from NuRadioReco.detector.webinterface.utils.helper_station import input_station_information, input_channel_information, test, build_collection_input
from NuRadioReco.utilities import units
from datetime import datetime

page_name = 'station'

def build_main_page(main_cont):
    main_cont.title('Build a station')
    main_cont.markdown(page_name)

    if 'collection_name' not in st.session_state:
        st.session_state['collection_name'] = ''

    # make it possible to select a collection/ create a collection
    if st.session_state.collection == '0':
        input_cont = main_cont.container()
        input_cont.subheader('Choose database collection')

        submit_button, collection_name = build_collection_input(input_cont)
        st.session_state.collection_name = collection_name
        if submit_button:
            st.session_state.collection = '1'
            st.experimental_rerun()

    # page to enter the station information
    if st.session_state.collection == '1':
        collection_name = st.session_state.collection_name
        print(collection_name)
        main_cont.info(f'You are now changing/writing to collection: {collection_name}')
        cont_warning_top = main_cont.container()
        main_cont.subheader('Input general station information')
        cont_station = main_cont.container()
        main_cont.subheader('Input channel information')
        cont_channel = main_cont.container()

        selected_station_name, selected_station_id, station_position, station_comm_date, station_decomm_date, sta_comment = input_station_information(cont_station, cont_warning_top, collection_name)
        selected_channel, selected_antenna_name, position_ant, ori_rot_ant, selected_antenna_type, comm_date_ant, decomm_date_ant, signal_chain_ant, cha_comment = input_channel_information(cont_channel, selected_station_id, collection_name)

        col1_button, col2_button = main_cont.columns([1,1])
        insert_button = col1_button.button('INSERT TO DB')
        back_button = col2_button.button('BACK TO COLLECTIONS')
        if back_button:
            st.session_state.collection = '0'
            st.experimental_rerun()

    #TODO if you update/overwrite a channel -> big warning with a extra button to click
    # if button inserted in the database -> some kind of motion to insure that channel is now in the database
    # --> use spinner?

# main page setup
page_configuration()

# initialize the session key (will be used to display different pages depending on the button clicked)
if 'key' not in st.session_state:
    st.session_state['key'] = '0'

# initialize the session key to change between input collection or station input (will be used to display different pages depending on the button clicked)
if 'collection' not in st.session_state:
    st.session_state['collection'] = '0'

main_container = st.container()  # container for the main part of the page (with all the input filed)
success_container = st.container()  # container to display the page when the data was submitted

if st.session_state.key == '0':
    build_main_page(main_container)  # after clicking the submit button, the session key is set to '1' and the page is rerun

if st.session_state.key == '1':
    build_success_page(success_container, page_name)  # after clicking the 'add another measurement' button, the session key is set to '0' and the page is rerun
