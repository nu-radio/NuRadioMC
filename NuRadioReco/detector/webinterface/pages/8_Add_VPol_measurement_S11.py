import streamlit as st
import pandas as pd
from plotly import subplots
import plotly.graph_objs as go
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page, create_single_plot, single_S_data_validation
from NuRadioReco.detector.webinterface.utils.helper_antenna import select_antenna_name, validate_global, insert_to_db
from NuRadioReco.detector.webinterface.utils.helper_protocol import load_measurement_protocols_from_db
from NuRadioReco.utilities import units

page_name = 'vpol'
s_name = 'S11'


def build_main_page(main_cont):
    main_cont.title('Add S11 measurement of VPol Antenna')
    main_cont.markdown(page_name)
    cont_warning_top = main_cont.container()

    # select the Antenna name
    VPol_name, VPol_dropdown, VPol_text = select_antenna_name(page_name, main_cont, cont_warning_top)

    # input of checkboxes, data_format, units and protocol
    working = main_cont.checkbox('channel is working', value=True)
    primary = main_cont.checkbox('Is this the primary measurement?', value=True)
    data_stat = main_cont.selectbox('specify the data format:', ['comma separated \",\"'])
    input_units = ['', '']
    col11, col22 = main_cont.columns([1, 1])
    input_units[0] = col11.selectbox('Units:', ['Hz', 'GHz', 'MHz'])
    input_units[1] = col22.selectbox('', ['VSWR','V', 'mV'])
    protocols_db = load_measurement_protocols_from_db()
    protocol = main_cont.selectbox('Specify the measurement protocol: (description of the protocols can be found [here](https://radio.uchicago.edu/wiki/index.php/Measurement_protocols))', protocols_db,
                                   help='Your measurement protocol is not listed? Please add it to the database [here](Add_measurement_protocol)')

    # upload the data
    uploaded_data = main_cont.file_uploader('Select File', accept_multiple_files=False)
    # container for warnings/infos at the botton
    cont_warning_bottom = main_cont.container()

    # input checks and enable the INSERT button
    S_data, sdata_validated = single_S_data_validation(cont_warning_bottom, uploaded_data, input_units)
    disable_button = validate_global(page_name, cont_warning_bottom, VPol_dropdown, VPol_text, working, sdata_validated, uploaded_data)
    figure = create_single_plot(s_name, cont_warning_bottom, S_data, 'frequency', 'magnitude', 'MHz', input_units[1])
    main_cont.plotly_chart(figure, use_container_width=True)

    # INSERT button
    upload_button = main_cont.button('INSERT TO DB', disabled=disable_button)

    # insert the data into the database and change to the success page by setting the session key
    if upload_button:
        insert_to_db(page_name, s_name, VPol_name, S_data, working, primary, protocol, input_units)
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
