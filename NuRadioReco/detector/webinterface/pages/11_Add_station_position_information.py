import copy
import time
import streamlit as st
import pandas as pd
from plotly import subplots
import plotly.graph_objs as go
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page
from NuRadioReco.detector.webinterface.utils.helper_station import insert_station_position_to_db, load_object_names
from NuRadioReco.utilities import units
from datetime import datetime
from datetime import time

page_name = 'station_position'


# TODO: include a check if the measurement time is within the commission time of the station
def validate_inputs(cont, measurement_name):
    validate_name = True
    if measurement_name != '':
        validate_name = False
    else:
        cont.error('No name entered for the measurement.')

    return validate_name


def insert_station_info(cont_station_warning, cont_station_info, selected_station_id, measurement_name, measurement_time, station_position, primary):
    if st.session_state.insert_station:
        # TODO:
        # station_info = load_station_infos(selected_station_id, page_name)
        station_info = {}
        if station_info != {}:
            cont_station_warning.warning('YOU ARE ABOUT TO CHANGE AN EXISTING STATION!')
            cont_station_warning.markdown('Do you really want to change an existing station?')
            col_b1, col_b2, phold = cont_station_warning.columns([0.2, 0.2, 1.6])
            yes_button = col_b1.button('YES')
            no_button = col_b2.button('NO')
            if yes_button:
                insert_station_position_to_db(selected_station_id, measurement_name, measurement_time, station_position, primary)
                st.session_state.insert_station = False
                st.session_state.station_success = True
                st.experimental_rerun()

            if no_button:
                st.session_state.insert_station = False
                st.session_state.station_success = False
                st.experimental_rerun()
        else:
            insert_station_position_to_db(selected_station_id, measurement_name, measurement_time, station_position, primary)
            st.session_state.insert_station = False
            st.session_state.station_success = True
            st.experimental_rerun()

    if st.session_state.station_success:
        cont_station_info.success('Station successfully added to the database!')
        st.session_state.station_success = False

def build_main_page(main_cont):
    main_cont.title('Add station position information')
    main_cont.markdown(page_name)

    if 'insert_station' not in st.session_state:
        st.session_state.insert_station = False

    # select a unique name for the measurement (survey_01, tape_measurement, ...)
    col1_name, col2_name = main_cont.columns([1, 1])
    db_measurement_names = load_object_names(page_name)
    print(db_measurement_names)
    db_measurement_names.insert(0, 'new measurement')
    selected_name = col1_name.selectbox('Select or enter a unique name for the measurement:', db_measurement_names)
    disabled_name_input = True
    if selected_name == 'new measurement':
        disabled_name_input = False
    name_input = col2_name.text_input('Select or enter a unique name for the measurement:', disabled=disabled_name_input, label_visibility='hidden')
    measurement_name = selected_name
    if measurement_name == 'new measurement':
        measurement_name = name_input

    # enter the information for the single stations
    cont_warning_top = main_cont.container()
    station_list = ['Station 11 (Nanoq)', 'Station 12 (Terianniaq)', 'Station 13 (Ukaleq)', 'Station 14 (Tuttu)', 'Station 15 (Umimmak)', 'Station 21 (Amaroq)', 'Station 22 (Avinngaq)',
                    'Station 23 (Ukaliatsiaq)', 'Station 24 (Qappik)', 'Station 25 (Aataaq)']
    selected_station = main_cont.selectbox('Select a station', station_list)
    # get the name and id out of the string
    selected_station_name = selected_station[selected_station.find('(') + 1:-1]
    selected_station_id = int(selected_station[len('Station '):len('Station ') + 2])

    # is primary measurement?
    primary = main_cont.checkbox('Is this the primary measurement?', value=True)

    # measurement time
    measurement_time = main_cont.date_input('Enter the time of the measurement:', value=datetime.utcnow(), help='The date when the measurement was conducted.')
    measurement_time = datetime.combine(measurement_time, time(0, 0, 0))

    # position
    col1_pos, col2_pos, col3_pos = main_cont.columns([1, 1, 1])
    # TODO load the position infos if already in the database
    # if station_info != {}:
    #     db_position = station_info[selected_station_id]['position']
    # else:
    #     db_position = [0, 0, 0]
    db_position = [0, 0, 0]
    x_pos = col1_pos.text_input('x position', value=db_position[0])
    y_pos = col2_pos.text_input('y position', value=db_position[1])
    z_pos = col3_pos.text_input('z position', value=db_position[2])
    position = [float(x_pos), float(y_pos), float(z_pos)]

    # TODO: check if already an entry for this measurement and station exists
    # # give a warning if the station already exist
    # if loaded_information != {}:
    #     cont_warning_top.warning(f'A database entry for the measurement '{measurement_name}' exits for the selected station: {selected_station}.')

    cont_station_warning = main_cont.container()
    disable_station_button = validate_inputs(cont_station_warning, measurement_name)

    insert_station = main_cont.button('INSERT STATION TO DB', disabled=disable_station_button)

    cont_station_info = main_cont.container()

    if insert_station:
        st.session_state.insert_station = True

    insert_station_info(cont_station_warning, cont_station_info, selected_station_id, measurement_name, measurement_time, position, primary)

# main page setup
page_configuration()

if 'station_success' not in st.session_state:
    st.session_state['station_success'] = False

# initialize the session key (will be used to display different pages depending on the button clicked)
if 'key' not in st.session_state:
    st.session_state['key'] = '0'

main_container = st.container()  # container for the main part of the page (with all the input filed)
success_container = st.container()  # container to display the page when the data was submitted

if st.session_state.key == '0':
    build_main_page(main_container)  # after clicking the submit button, the session key is set to '1' and the page is rerun

if st.session_state.key == '1':
    build_success_page(success_container, page_name)  # after clicking the 'add another measurement' button, the session key is set to '0' and the page is rerun
