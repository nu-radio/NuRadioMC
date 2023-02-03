import copy
import time
import streamlit as st
import pandas as pd
from plotly import subplots
import plotly.graph_objs as go
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page
from NuRadioReco.detector.webinterface.utils.helper_station import input_station_information, input_channel_information, input_device_information, build_collection_input, validate_station_inputs, \
    validate_channel_inputs, validate_device_inputs, insert_general_station_info_to_db, insert_general_channel_info_to_db, insert_general_device_info_to_db
from NuRadioReco.utilities import units
from datetime import datetime

page_name = 'station'
collection_name = 'station_rnog'


def insert_station_info(cont_station_warning, cont_station_info, selected_station_id, collection_name, selected_station_name, sta_comment, station_comm_date, station_decomm_date, station_info):
    if st.session_state.insert_station:
        if station_info != {}:
            cont_station_warning.warning('YOU ARE ABOUT TO CHANGE AN EXISTING STATION!')
            cont_station_warning.markdown('Do you really want to change an existing station?')
            col_b1, col_b2, phold = cont_station_warning.columns([0.2, 0.2, 1.6])
            yes_button = col_b1.button('YES')
            no_button = col_b2.button('NO')
            if yes_button:
                insert_general_station_info_to_db(selected_station_id, collection_name, selected_station_name, sta_comment, station_comm_date, station_decomm_date)
                st.session_state.insert_station = False
                st.session_state.station_success = True
                st.experimental_rerun()

            if no_button:
                st.session_state.insert_station = False
                st.session_state.station_success = False
                st.experimental_rerun()
        else:
            insert_general_station_info_to_db(selected_station_id, collection_name, selected_station_name, sta_comment, station_comm_date, station_decomm_date)
            st.session_state.insert_station = False
            st.session_state.station_success = True
            st.experimental_rerun()

    if st.session_state.station_success:
        cont_station_info.success('Station successfully added to the database!')
        st.session_state.station_success = False


def insert_channel_info(collection_name, warning_cont, info_cont, selected_station_id, selected_channel, signal_chain, ant_name, channel_type, channel_comment, commission_time, decommission_time, station_info):
    if st.session_state.insert_channel:
        if station_info != {}:
            channel_info = station_info[selected_station_id]['channels']
            if selected_channel in channel_info.keys():
                warning_cont.warning('YOU ARE ABOUT TO CHANGE AN EXISTING CHANNEL!')
                warning_cont.markdown('Do you really want to change an existing channel?')
                col_b1, col_b2, phold = warning_cont.columns([0.2, 0.2, 1.6])
                yes_button = col_b1.button('YES')
                no_button = col_b2.button('NO')
                if yes_button:
                    insert_general_channel_info_to_db(selected_station_id, collection_name, selected_channel, signal_chain, ant_name, channel_type, channel_comment, commission_time, decommission_time)
                    st.session_state.insert_channel = False
                    st.session_state.channel_success = True
                    st.experimental_rerun()

                if no_button:
                    st.session_state.insert_channel = False
                    st.session_state.channel_success = False
                    st.experimental_rerun()
            else:
                # information will be inserted into the database, without requiring any action
                insert_general_channel_info_to_db(selected_station_id, collection_name, selected_channel, signal_chain, ant_name, channel_type, channel_comment, commission_time, decommission_time)
                st.session_state.insert_channel = False
                st.session_state.channel_success = True

    if st.session_state.channel_success:
        info_cont.success('Channel successfully added to the database!')
        st.session_state.channel_success = False

        # if there is no corresponding station in the database -> the insert button is disabled (see validate_channel_inputs())


def insert_device_info(collection_name, warning_cont, info_cont, selected_station_id, selected_device_id, device_name, amp_name, device_comment, commission_time, decommission_time, station_info):
    if st.session_state.insert_device:
        if station_info != {}:
            device_info = station_info[selected_station_id]['devices']
            if selected_device_id in device_info.keys():
                warning_cont.warning('YOU ARE ABOUT TO CHANGE AN EXISTING DEVICE!')
                warning_cont.markdown('Do you really want to change an existing device?')
                col_d1, col_d2, phold = warning_cont.columns([0.2, 0.2, 1.6])
                yes_button = col_d1.button('YES')
                no_button = col_d2.button('NO')
                if yes_button:
                    insert_general_device_info_to_db(selected_station_id, collection_name, selected_device_id, device_name, amp_name, device_comment, commission_time, decommission_time)
                    st.session_state.insert_device = False
                    st.session_state.device_success = True
                    st.experimental_rerun()

                if no_button:
                    st.session_state.insert_device = False
                    st.session_state.device_success = False
                    st.experimental_rerun()
            else:
                # information will be inserted into the database, without requiring any action
                insert_general_device_info_to_db(selected_station_id, collection_name, selected_device_id, device_name, amp_name, device_comment, commission_time, decommission_time)
                st.session_state.insert_device = False
                st.session_state.device_success = True

    if st.session_state.device_success:
        info_cont.success('Device successfully added to the database!')
        st.session_state.device_success = False


def build_main_page(main_cont):
    main_cont.title('Add general station information')
    main_cont.markdown(page_name)

    if 'insert_station' not in st.session_state:
        st.session_state.insert_station = False

    if 'insert_channel' not in st.session_state:
        st.session_state.insert_channel = False

    if 'insert_device' not in st.session_state:
        st.session_state.insert_device = False

    # page to enter the station information
    cont_warning_top = main_cont.container()
    main_cont.subheader('Input general station information')
    cont_station = main_cont.container()
    selected_station_name, selected_station_id, station_comm_date, station_decomm_date, sta_comment, station_info = input_station_information(cont_station, cont_warning_top, collection_name)

    cont_station_info = cont_station.container()
    disable_station_button = validate_station_inputs(cont_station_info, station_comm_date, station_decomm_date)

    insert_station = cont_station.button('INSERT STATION TO DB', disabled=disable_station_button)
    cont_station_warning = cont_station.container()

    if insert_station:
        st.session_state.insert_station = True

    insert_station_info(cont_station_warning, cont_station_info, selected_station_id, collection_name, selected_station_name, sta_comment, station_comm_date, station_decomm_date, station_info)

    main_cont.subheader('Input channel information')
    cont_channel = main_cont.container()

    selected_channel, selected_antenna_name, selected_antenna_type, comm_date_ant, decomm_date_ant, signal_chain_ant, cha_comment, function_channel = input_channel_information(cont_channel, selected_station_id,
                                                                                                                                                                                collection_name, station_info)

    # container for warnings/infos at the botton
    cont_warning_bottom = main_cont.container()

    disable_insert_button = validate_channel_inputs(collection_name, cont_warning_bottom, selected_station_name, comm_date_ant, decomm_date_ant, selected_channel, signal_chain_ant)

    insert_channel = main_cont.button('INSERT CHANNEL TO DB', disabled=disable_insert_button)

    cont_channel_warning = main_cont.container()

    if insert_channel:
        st.session_state.insert_channel = True
    insert_channel_info(collection_name, cont_channel_warning, cont_warning_bottom, selected_station_id, selected_channel, signal_chain_ant, selected_antenna_name, selected_antenna_type, cha_comment,
                        comm_date_ant, decomm_date_ant, station_info)

    main_cont.subheader('Input device information')
    cont_device = main_cont.container()

    device_id, device_name, comm_date_dev, decomm_date_dev, dev_comment, function_device, amp_name = input_device_information(cont_device, selected_station_id, station_info)

    # container for warnings/infos at the botton
    cont_device_warning_bottom = main_cont.container()

    disable_device_button = validate_device_inputs(cont_device_warning_bottom, selected_station_name, comm_date_dev, decomm_date_dev, device_id)

    insert_device = main_cont.button('INSERT DEVICE TO DB', disabled=disable_device_button)

    cont_device_warning = main_cont.container()
    if insert_device:
        st.session_state.insert_device = True

    insert_device_info('station_rnog', cont_device_warning, cont_device_warning_bottom, selected_station_id, device_id, device_name, amp_name, dev_comment, comm_date_dev, decomm_date_dev, station_info)


# main page setup
page_configuration()

if 'station_success' not in st.session_state:
    st.session_state['station_success'] = False

if 'channel_success' not in st.session_state:
    st.session_state['channel_success'] = False

if 'device_success' not in st.session_state:
    st.session_state['device_success'] = False

# initialize the session key (will be used to display different pages depending on the button clicked)
if 'key' not in st.session_state:
    st.session_state['key'] = '0'

main_container = st.container()  # container for the main part of the page (with all the input filed)
success_container = st.container()  # container to display the page when the data was submitted

if st.session_state.key == '0':
    build_main_page(main_container)  # after clicking the submit button, the session key is set to '1' and the page is rerun

if st.session_state.key == '1':
    build_success_page(success_container, page_name)  # after clicking the 'add another measurement' button, the session key is set to '0' and the page is rerun
