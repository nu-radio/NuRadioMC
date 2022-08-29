import copy

import streamlit as st
import pandas as pd
from plotly import subplots
import plotly.graph_objs as go
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page, select_downhole, single_S_data_validation, create_ten_plots, validate_global_downhole, insert_downhole_to_db, read_measurement_time
from NuRadioReco.utilities import units

page_name = 'downhole'
s_name = ['S11', 'S12', 'S21', 'S22']


def build_main_page(main_cont):
    main_cont.title('Add S parameter measurements for the FULL DOWNHOLE CHAIN')
    main_cont.markdown(page_name)
    cont_warning_top = main_cont.container()

    # select the cable
    downhole_name, downhole_dropdown, breakout_id, breakout_name, iglu_id, drap_id, temp = select_downhole(page_name, main_cont, cont_warning_top)

    # input of checkboxes, data_format, units and protocol
    working = main_cont.checkbox('channel is working', value=True)
    primary = main_cont.checkbox('Is this the primary measurement?', value=True)
    data_stat = main_cont.selectbox('specify the data format:', ['comma separated \",\"'])
    input_units = ['', '', '']
    col11, col22, col33 = main_cont.columns([1, 1, 1])
    input_units[0] = col11.selectbox('Units:', ['Hz', 'GHz', 'MHz'])
    input_units[1] = col22.selectbox('', ['MAG','V', 'mV'])
    input_units[2] = col33.selectbox('', ['deg', 'rad'])
    protocol = main_cont.selectbox('Specify the measurement protocol:', ['Chicago2020', 'Erlangen2020', 'Chicago2022', 'Erlangen2022'])
    group_delay = main_cont.number_input('Enter group delay correction [ns] at around 200 MHz:', value=0, step=1, help='Read off the group delay from the left group delay plot below (after inserting data) and input the result here. A plot for the group delay corrected plot will be shown below on the right.')
    group_delay_arr = [0, 0, group_delay, 0]
    # upload the data
    uploaded_data = main_cont.file_uploader('Select your measurement:', accept_multiple_files=False)
    uploaded_data_copy = copy.deepcopy(uploaded_data) # copy needed to extract the measurement time
    # container for warnings/infos at the botton
    cont_warning_bottom = main_cont.container()

    # # input checks and enable the INSERT button
    S_data, sdata_validated = single_S_data_validation(cont_warning_bottom, uploaded_data, input_units)
    disable_button = validate_global_downhole(page_name, cont_warning_bottom, downhole_dropdown, downhole_name, breakout_id, breakout_name, working, sdata_validated, uploaded_data)
    figure = create_ten_plots(s_name, cont_warning_bottom, S_data, ['frequency'], ['group delay', 'magnitude', 'phase'], input_units, group_delay)
    main_cont.plotly_chart(figure, use_container_width=True)

    # INSERT button
    upload_button = main_cont.button('INSERT TO DB', disabled=disable_button)

    # insert the data into the database and change to the success page by setting the session key
    if upload_button:
        measure_time = read_measurement_time(uploaded_data_copy)
        insert_downhole_to_db(page_name, s_name, downhole_name, S_data, input_units, working, primary, protocol, breakout_id, breakout_name, iglu_id, drap_id, temp, measure_time, group_delay_arr)
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
