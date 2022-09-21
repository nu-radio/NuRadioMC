import copy
import time
import streamlit as st
import pandas as pd
from plotly import subplots
import plotly.graph_objs as go
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page
from NuRadioReco.detector.webinterface.utils.helper_station import input_station_information, input_channel_information, build_collection_input, validate_station_inputs, validate_channel_inputs, insert_station_to_db, insert_channel_to_db, load_station_infos
from NuRadioReco.utilities import units
from datetime import datetime

page_name = 'station'


def insert_station_info(cont_station_warning, cont_station_info, selected_station_id, collection_name, selected_station_name, station_position, sta_comment, station_comm_date, station_decomm_date):
    if st.session_state.insert_station:
        station_info = load_station_infos(selected_station_id, collection_name)
        if station_info != {}:
            cont_station_warning.warning('YOU ARE ABOUT TO CHANGE AN EXISTING STATION!')
            cont_station_warning.markdown('Do you really want to change an existing station?')
            col_b1, col_b2, phold = cont_station_warning.columns([0.2, 0.2, 1.6])
            yes_button = col_b1.button('YES')
            no_button = col_b2.button('NO')
            if yes_button:
                insert_station_to_db(selected_station_id, collection_name, selected_station_name, station_position, sta_comment, station_comm_date, station_decomm_date)
                st.session_state.insert_station = False
                st.session_state.station_success = True
                st.experimental_rerun()

            if no_button:
                st.session_state.insert_station = False
                st.session_state.station_success = False
                st.experimental_rerun()
        else:
            insert_station_to_db(selected_station_id, collection_name, selected_station_name, station_position, sta_comment, station_comm_date, station_decomm_date)
            st.session_state.insert_station = False
            st.session_state.station_success = True
            st.experimental_rerun()

    if st.session_state.station_success:
        cont_station_info.success('Station successfully added to the database!')
        st.session_state.station_success = False


def build_main_page(main_cont):
    main_cont.title('Build a station')
    main_cont.markdown(page_name)

    if 'insert_station' not in st.session_state:
        st.session_state.insert_station = False

    if 'insert_channel' not in st.session_state:
        st.session_state.insert_channel = False

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
        main_cont.info(f'You are now changing/writing to collection: {collection_name}')
        back_button = main_cont.button('BACK TO COLLECTIONS')
        if back_button:
            st.session_state.collection = '0'
            st.experimental_rerun()

        cont_warning_top = main_cont.container()
        main_cont.subheader('Input general station information')
        cont_station = main_cont.container()
        selected_station_name, selected_station_id, station_position, station_comm_date, station_decomm_date, sta_comment = input_station_information(cont_station, cont_warning_top, collection_name)

        cont_station_info = cont_station.container()
        disable_station_button = validate_station_inputs(cont_station_info, station_comm_date, station_decomm_date)

        insert_station = cont_station.button('INSERT STATION TO DB', disabled=disable_station_button)
        cont_station_warning = cont_station.container()

        if insert_station:
            st.session_state.insert_station = True

        insert_station_info(cont_station_warning, cont_station_info, selected_station_id, collection_name, selected_station_name, station_position, sta_comment, station_comm_date, station_decomm_date)

        main_cont.subheader('Input channel information')
        cont_channel = main_cont.container()

        selected_channel, selected_antenna_name, position_ant, ori_rot_ant, selected_antenna_type, comm_date_ant, decomm_date_ant, signal_chain_ant, cha_comment, function_channel = input_channel_information(cont_channel, selected_station_id, collection_name)

        # container for warnings/infos at the botton
        cont_warning_bottom = main_cont.container()

        disable_insert_button = validate_channel_inputs(collection_name, cont_warning_bottom, selected_station_name, comm_date_ant, decomm_date_ant, selected_channel, signal_chain_ant)

        insert_channel = main_cont.button('INSERT CHANNEL TO DB', disabled=disable_insert_button)

        if insert_channel:
            st.session_state.insert_channel = True
        if st.session_state.insert_channel:
            station_info = load_station_infos(selected_station_id, collection_name)
            if station_info != {}:
                channel_info = station_info[selected_station_id]['channels']
                if selected_channel in channel_info.keys():
                    # there is already an existing one
                    # decommission first and then update
                    pass
                else:
                    # insert channel to db
                    # insert_channel_to_db(selected_station_id, collection_name, selected_station_name, station_position, sta_comment, station_comm_date, station_decomm_date, selected_channel)
                    pass
            else:
                print('THERE IS NO FITTING STATION IN THE DB')
                # maybe put disable insert button and print error

# main page setup
page_configuration()

if 'station_success' not in st.session_state:
    st.session_state['station_success'] = False

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

print(st.session_state)