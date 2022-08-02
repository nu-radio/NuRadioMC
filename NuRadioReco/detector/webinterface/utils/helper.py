import streamlit as st
from plotly import subplots
import plotly.graph_objs as go
import pandas as pd
from NuRadioReco.detector.webinterface.utils.units_helper import str_to_unit
from NuRadioReco.utilities import units
import numpy as np
from NuRadioReco.detector.detector_mongo import Detector

det = Detector(database_connection='test')


def build_success_page(cont_success, measurement_name):
    cont_success.success(f'{measurement_name} was successfully added to the data base.')
    if cont_success.button('Add another measurement'):
        cont_success.empty()
        st.session_state.key = '0'
        st.experimental_rerun()
    st.balloons()


def Sdata_validation(container_bottom, uploaded_data, input_units, additional_information=""):
    Sdata_validate = False
    if uploaded_data is None:
        if additional_information == '':
            container_bottom.info('No data selected')
        else:
            container_bottom.info(f'No {additional_information} data selected')
        Sdata_validate = False
        return [[],[]], Sdata_validate
    else:
        try:
            SData = convert_uploaded_data(uploaded_data, input_units)
            Sdata_validate = True
            return SData, Sdata_validate
        except:
            Sdata_validate = False
            return [[],[]], Sdata_validate


def convert_uploaded_data(uploaded_data_file, input_units):
    data_x = []
    data_y = []
    if uploaded_data_file is not None:
        # load the uploaded data as a pd.dataframe and extract the x,y values
        data_frame = pd.read_csv(uploaded_data_file)
        data_x_index = data_frame.index
        data_y_values = data_frame[data_frame.keys()[0]].values

        # x,y are given as strings. Need to convert them to float
        for dx, dy in zip(data_x_index, data_y_values):
            try:
                # check if the conversion works
                help_x = float(dx)
            except:
                # if not, there might be text in this entry. We skip the entry.
                pass
            else:
                data_x.append(float(dx) * str_to_unit[input_units[0]])
                data_y.append(float(dy) * str_to_unit[input_units[1]])
    return [data_x, data_y]


# HPol and VPol

def create_single_plot(S_names, plot_info_cont, data_x, data_y, xlabel, ylabel, xunit, yunit):
    fig = subplots.make_subplots(rows=1, cols=1)
    if data_x != [] and data_y != []:
        fig.append_trace(go.Scatter(x=np.asarray(data_x) / str_to_unit[xunit], y=np.asarray(data_y) / str_to_unit[yunit], opacity=0.7, marker={'color': "blue", 'line': {'color': "blue"}}, name='magnitude'), 1, 1)
        fig['layout']['xaxis1'].update(title=f'{xlabel} [{xunit}]')
        fig['layout']['yaxis1'].update(title=f'{ylabel} [{yunit}]')
        plot_info_cont.info(f'you entered {len(data_x)} {xlabel} from {min(data_x) / str_to_unit[xunit]:.4g}{xunit} to {max(data_x)/str_to_unit[xunit]:.4g}{xunit} and {S_names} {ylabel} {len(data_y)} values within the range of {min(data_y)/str_to_unit[yunit]:.4g}{yunit} to {max(data_y)/str_to_unit[yunit]:.4g}{yunit}')

    return fig


def select_antenna_name(antenna_type, container, warning_container):
    selected_antenna_name = ''
    # update the dropdown menu
    antenna_names = det.get_Antenna_names(antenna_type)
    col1, col2 = container.columns([1, 1])
    antenna_dropdown = col1.selectbox('Select existing antenna or enter unique name of new antenna:', antenna_names)
    # checking which antenna is selected
    if antenna_dropdown == f'new {antenna_type}':
        # if new antenna is chosen: The text input will be enabled
        disable_new_antenna_name = False
    else:
        # Otherwise the chosen antenna will be used and warning will be given
        disable_new_antenna_name = True
        selected_antenna_name = antenna_dropdown
        warning_container.warning(f'You are about to override the {antenna_type} unit {antenna_dropdown}!')
    antenna_text_input = col2.text_input('', placeholder=f'new unique {antenna_type} name', disabled=disable_new_antenna_name, help='This is to explain what to do')
    if antenna_dropdown == f'new {antenna_type}':
        selected_antenna_name = antenna_text_input

    return selected_antenna_name, antenna_dropdown, antenna_text_input


def validate_global(page_name, container_bottom, antenna_name, new_antenna_name, channel_working, Sdata_validated, uploaded_data):
    disable_insert_button = True
    name_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if antenna_name == '':
        container_bottom.error('Antenna name is not set')
    elif antenna_name == f'new {page_name}' and (new_antenna_name is None or new_antenna_name == ''):
        container_bottom.error(f'Antenna name dropdown is set to \'new {page_name}\', but no new antenna name was entered.')
    else:
        name_validation = True

    if name_validation:
        if not Sdata_validated and uploaded_data is not None:
            container_bottom.error('There is a problem with the input data')
            disable_insert_button = True
        elif Sdata_validated:
            disable_insert_button = False
            container_bottom.success('Input fields are validated')

        if not channel_working:
            container_bottom.warning('The channel is set to not working')
            disable_insert_button = False
            container_bottom.success('Input fields are validated')
    else:
        disable_insert_button = True

    return disable_insert_button


def insert_to_db(page_name, s_name, antenna_name, data, working, primary, protocol, input_units):
    if not working:
        det.set_not_working(page_name, antenna_name)
    else:
        if primary and antenna_name in det.get_Antenna_names(page_name):
            det.update_primary(page_name, antenna_name)
        det.antenna_add_Sparameter(page_name, antenna_name, s_name, data, primary, protocol, input_units)


# SURFACE CABLE

def select_cable(page_name, main_container, warning_container_top):
    col1_cable, col2_cable, col3_cable = main_container.columns([1,1,1])
    cable_types = []
    cable_stations = []
    cable_channels = []
    if page_name == 'surfCABLE':
        cable_types=['Choose an option', '11 meter signal']
        cable_stations = ['Choose an option', 'Station 1 (11 Nanoq)', 'Station 2 (12 Terianniaq)', 'Station 3 (13 Ukaleq)', 'Station 4 (14 Tuttu)', 'Station 5 (15 Umimmak)', 'Station 6 (21 Amaroq)', 'Station 7 (22 Avinngaq)', 'Station 8 (23 Ukaliatsiaq)', 'Station 9 (24 Qappik)','Station 10 (25 Aataaq)']
        cable_channels = ['Choose an option', 'Channel 1 (0)', 'Channel 2 (1)', 'Channel 3 (2)', 'Channel 4 (3)', 'Channel 5 (4)', 'Channel 6 (5)', 'Channel 7 (6)', 'Channel 8 (7)', 'Channel 9 (8)']
    elif page_name == 'CABLE':
        cable_types = ['Choose an option', 'Orange (1m)', 'Blue (2m)', 'Green (3m)', 'White (4m)', 'Brown (5m)', 'Red/Grey (6m)']
        cable_stations = ['Choose an option', 'Station 11 (Nanoq)', 'Station 12 (Terianniaq)', 'Station 13 (Ukaleq)', 'Station 14 (Tuttu)', 'Station 15 (Umimmak)', 'Station 21 (Amaroq)',
                          'Station 22 (Avinngaq)', 'Station 23 (Ukaliatsiaq)', 'Station 24 (Qappik)', 'Station 25 (Aataaq)']
        cable_channels = ['Choose an option', 'A (Power String)', 'B (Helper String)', 'C (Helper String)']

    cable_type = col1_cable.selectbox('Select existing cable :', cable_types)
    cable_station = col2_cable.selectbox('', cable_stations)
    cable_channel = col3_cable.selectbox('', cable_channels)

    cable_name = ""
    if page_name == 'surfCABLE':
        cable_name = cable_station[cable_station.find('(')+1:cable_station.find('(')+3] + cable_channel[cable_channel.find('(')+1:cable_channel.rfind(')')] + cable_type[:cable_type.find(' meter')]
    elif page_name == 'CABLE':
        cable_name = cable_station[len('stations'): len('stations') + 2] + cable_channel[:1] + cable_type[cable_type.find('(')+1:cable_type.find(')')-1]

    if cable_name in det.get_cable_names(page_name):
        if page_name == 'surfCABLE':
            warning_container_top.warning(f'You are about to override the {page_name} unit \'{cable_name[-2:]} meter, station {cable_name[:2]}, channel {cable_name[2:-2]}\'!')
        elif page_name == 'CABLE':
            warning_container_top.warning(f'You are about to override the {page_name} unit \'{cable_name[-1:]} meter, station {cable_name[:2]}, string {cable_name[2:-1]}\'!')


    return cable_type, cable_station, cable_channel, cable_name


def validate_global_cable(container_bottom, cable_type, cable_sta, cable_cha, channel_working, Sdata_validated_magnitude, Sdata_validated_phase, uploaded_data_magnitude, uploaded_data_phase):
    disable_insert_button = True
    name_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if cable_type == 'Choose an option' or cable_sta == 'Choose an option' or cable_cha == 'Choose an option':
        container_bottom.error('Not all cable options are selected')
        name_validation = False
    else:
        name_validation = True

    if name_validation:
        if not Sdata_validated_magnitude and uploaded_data_magnitude is not None:
            container_bottom.error('There is a problem with the magnitude input data')
            disable_insert_button = True

        if not Sdata_validated_phase and uploaded_data_phase is not None:
            container_bottom.error('There is a problem with the phase input data')
            disable_insert_button = True

        if Sdata_validated_magnitude and Sdata_validated_phase:
            disable_insert_button = False
            container_bottom.success('All inputs validated')

        if not channel_working:
            container_bottom.warning('The channel is set to not working')
            disable_insert_button = False
            container_bottom.success('All inputs validated')
    else:
        disable_insert_button = True

    return disable_insert_button


def create_double_plot(S_names, plot_info_cont, data1, data2, labels1, labels2, units1, units2):
    fig = subplots.make_subplots(rows=1, cols=2)
    if data1[0] != [] and data1[1] != []:
        fig.append_trace(go.Scatter(x=np.asarray(data1[0]) / str_to_unit[units1[0]], y=np.asarray(data1[1]) / str_to_unit[units1[1]], opacity=0.7, marker={'color': "blue", 'line': {'color': "blue"}}, name='magnitude'), 1, 1)
        fig['layout']['xaxis1'].update(title=f'{labels1[0]} [{units1[0]}]')
        fig['layout']['yaxis1'].update(title=f'{labels1[1]} [{units1[1]}]')
        plot_info_cont.info(f'you entered {len(data1[0])} {labels1[0]} from {min(data1[0]) / str_to_unit[units1[0]]:.4g}{units1[0]} to {max(data1[0])/str_to_unit[units1[0]]:.4g}{units1[0]} and {S_names} {labels1[1]} {len(data1[1])} values within the range of {min(data1[1])/str_to_unit[units1[1]]:.4g}{units1[1]} to {max(data1[1])/str_to_unit[units1[1]]:.4g}{units1[1]}')
    if data2[0] != [] and data2[1] != []:
        fig.append_trace(go.Scatter(x=np.asarray(data2[0]) / str_to_unit[units2[0]], y=np.asarray(data2[1]) / str_to_unit[units2[1]], opacity=0.7, marker={'color': "red", 'line': {'color': "red"}}, name='phase'), 1, 2)
        fig['layout']['xaxis2'].update(title=f'{labels2[0]} [{units2[0]}]')
        fig['layout']['yaxis2'].update(title=f'{labels2[1]} [{units2[1]}]')
        plot_info_cont.info(f'you entered {len(data2[0])} {labels2[0]} from {min(data2[0]) / str_to_unit[units2[0]]:.4g}{units2[0]} to {max(data2[0])/str_to_unit[units2[0]]:.4g}{units2[0]} and {S_names} {labels2[1]} {len(data2[1])} values within the range of {min(data2[1])/str_to_unit[units2[1]]:.4g}{units2[1]} to {max(data2[1])/str_to_unit[units2[1]]:.4g}{units2[1]}')

    return fig


def insert_cable_to_db(page_name, s_name, cable_name, data_m, data_p, input_units, working, primary, protocol):
    if not working:
        det.set_not_working(page_name, cable_name)
    else:
        if primary and cable_name in det.get_cable_names(page_name):
            det.update_primary(page_name, cable_name)
        det.cable_add_Sparameter(page_name, cable_name, s_name, data_m, data_p, input_units, primary, protocol)


# IGLU

def select_IGLU(page_name, main_container, warning_container_top):
    col1_I, col2_I, col3_I, col4_I, col5_I = main_container.columns([1,1,1,1,1])

    disable_new_input = True
    selected_board = col1_I.selectbox('Select existing board or enter unique name of new board:', ['new board'])
    new_board_name = col2_I.text_input('', placeholder=f'new unique board name', disabled=disable_new_input)
    new_laser_serial = col3_I.text_input('', placeholder=f'laser_serial', disabled=disable_new_input)
    selected_DRAB = col4_I.selectbox('', ['without DRAB'])
    selected_Temp = col5_I.selectbox('', ['room temp (20°C'])

    return ''
