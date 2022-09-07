import streamlit as st
from plotly import subplots
import plotly.graph_objs as go
import pandas as pd
from NuRadioReco.detector.webinterface.utils.units_helper import str_to_unit
from NuRadioReco.utilities import units
import numpy as np
from NuRadioReco.detector.detector_mongo import Detector
from datetime import datetime

det = Detector(database_connection='test')

# GENERAL

def build_success_page(cont_success, measurement_name):
    cont_success.success(f'{measurement_name} was successfully added to the data base.')
    # if the button is pressed the code is rerun from the top
    if cont_success.button('Add another measurement'):
        cont_success.empty()
    del st.session_state['key']
    st.balloons()


def single_S_data_validation(container_bottom, uploaded_data, input_units, additional_information=""):
    Sdata_validate = False
    if uploaded_data is None:
        if additional_information == '':
            container_bottom.info('No data selected')
        else:
            container_bottom.info(f'No {additional_information} data selected')
        Sdata_validate = False
        return [], Sdata_validate
    else:
        try:
            SData = read_uploaded_S_data(uploaded_data, input_units)
            Sdata_validate = True
            return SData, Sdata_validate
        except:
            Sdata_validate = False
            return [], Sdata_validate


# def read_uploaded_single_S_data(uploaded_data_file, input_units):
#     data_x = []
#     data_y = []
#     if uploaded_data_file is not None:
#         # load the uploaded data as a pd.dataframe and extract the x,y values
#         data_frame = pd.read_csv(uploaded_data_file)
#         data_x_index = data_frame.index
#         data_y_values = data_frame[data_frame.keys()[0]].values
#
#         # x,y are given as strings. Need to convert them to float
#         for dx, dy in zip(data_x_index, data_y_values):
#             try:
#                 # check if the conversion works
#                 help_x = float(dx)
#             except:
#                 # if not, there might be text in this entry. We skip the entry.
#                 pass
#             else:
#                 data_x.append(float(dx) * str_to_unit[input_units[0]])
#                 data_y.append(float(dy) * str_to_unit[input_units[1]])
#     return [data_x, data_y]


def read_uploaded_S_data(uploaded_data_file, input_units):
    data_converted = []
    if uploaded_data_file is not None:
        print(uploaded_data_file)
        # load the uploaded data as a pd.dataframe and extract the x,y values
        data = np.loadtxt(uploaded_data_file, dtype='float', delimiter=',', comments=['BEGIN', 'END', '!', 'Freq'])
        data_converted = []
        for i in range(len(data[0])):
            unit_name=''
            if i == 0:
                unit_name = input_units[0]
            else:
                unit_name = input_units[(i+1)%2 +1]

            help_data = data[:,i] * str_to_unit[unit_name]
            data_converted.append(help_data)

    return data_converted


def read_measurement_time(uploaded_data_file):
    measurement_time = ''
    if uploaded_data_file is not None:
        data = np.loadtxt(uploaded_data_file, dtype='str',skiprows=1, usecols=(0,1,2), delimiter=',', max_rows=2)
        for rows in data:
            if '!Date' in rows[0]:
                m_time = rows[0][len('!Date: '):] + rows[1] + rows[2]
        measurement_time = datetime.strptime(m_time, '%A %B %d %Y %X')

    return measurement_time


# HPol and VPol

def create_single_plot(S_names, plot_info_cont, data, xlabel, ylabel, xunit, yunit):
    fig = subplots.make_subplots(rows=1, cols=1)
    if data != [] :
        data_x = data[0]
        data_y = data[1]
        fig.append_trace(go.Scatter(x=np.asarray(data_x) / str_to_unit[xunit], y=np.asarray(data_y) / str_to_unit[yunit], opacity=0.7, marker={'color': "blue", 'line': {'color': "blue"}}, name='magnitude'), 1, 1)
        fig['layout']['xaxis1'].update(title=f'{xlabel} [{xunit}]')
        fig['layout']['yaxis1'].update(title=f'{ylabel} [{yunit}]')
        plot_info_cont.info(f'you entered {len(data_x)} {xlabel} from {min(data_x) / str_to_unit[xunit]:.4g}{xunit} to {max(data_x)/str_to_unit[xunit]:.4g}{xunit} and {S_names} {ylabel} {len(data_y)} values within the range of {min(data_y)/str_to_unit[yunit]:.4g}{yunit} to {max(data_y)/str_to_unit[yunit]:.4g}{yunit}')

    return fig


def create_double_plot(S_names, plot_info_cont, data1, data2, labels1, labels2, units1, units2):
    fig = subplots.make_subplots(rows=1, cols=2)
    if data1 != []:
        data1_x = data1[0]
        data1_y = data1[1]
        fig.append_trace(go.Scatter(x=np.asarray(data1_x) / str_to_unit[units1[0]], y=np.asarray(data1_y) / str_to_unit[units1[1]], opacity=0.7, marker={'color': "blue", 'line': {'color': "blue"}}, name='magnitude'), 1, 1)
        fig['layout']['xaxis1'].update(title=f'{labels1[0]} [{units1[0]}]')
        fig['layout']['yaxis1'].update(title=f'{labels1[1]} [{units1[1]}]')
        plot_info_cont.info(f'you entered {len(data1_x)} {labels1[0]} from {min(data1_x) / str_to_unit[units1[0]]:.4g}{units1[0]} to {max(data1_x)/str_to_unit[units1[0]]:.4g}{units1[0]} and {S_names} {labels1[1]} {len(data1[1])} values within the range of {min(data1_y)/str_to_unit[units1[1]]:.4g}{units1[1]} to {max(data1_y)/str_to_unit[units1[1]]:.4g}{units1[1]}')
    if data2 != []:
        data2_x = data2[0]
        data2_y = data2[1]
        fig.append_trace(go.Scatter(x=np.asarray(data2_x) / str_to_unit[units2[0]], y=np.asarray(data2_y) / str_to_unit[units2[1]], opacity=0.7, marker={'color': "red", 'line': {'color': "red"}}, name='phase'), 1, 2)
        fig['layout']['xaxis2'].update(title=f'{labels2[0]} [{units2[0]}]')
        fig['layout']['yaxis2'].update(title=f'{labels2[1]} [{units2[1]}]')
        plot_info_cont.info(f'you entered {len(data2_x)} {labels2[0]} from {min(data2_x) / str_to_unit[units2[0]]:.4g}{units2[0]} to {max(data2_x)/str_to_unit[units2[0]]:.4g}{units2[0]} and {S_names} {labels2[1]} {len(data2_y)} values within the range of {min(data2_y)/str_to_unit[units2[1]]:.4g}{units2[1]} to {max(data2_y)/str_to_unit[units2[1]]:.4g}{units2[1]}')

    return fig


def create_ten_plots(S_names, plot_info_cont, data, xlabels, ylabels, input_units, groupdelay_correction):
    if data != []:
        phase = data[6]
        freq = data[0]
        delta_freq = freq[1] - freq[0]
        phase_corr = phase + groupdelay_correction * freq * 2 * np.pi
        phase_corr_0 = phase + 0 * freq * 2 * np.pi
        calc_corr_group_delay = -np.diff(np.unwrap(phase_corr)) / delta_freq / 2 / np.pi
        calc_corr_group_delay_0 = -np.diff(np.unwrap(phase_corr_0)) / delta_freq / 2 / np.pi

        frequency_unit = 'MHz'
        group_delay_unit = 'ns'
        subtitles = ["Group Delay", "Corrected Group Delay"]
        for snames in S_names:
            subtitles.append(f'{snames} magnitude')
            subtitles.append(f'{snames} phase')

        fig = subplots.make_subplots(rows=6, cols=2,
                                     specs=[[{"rowspan": 2}, {"rowspan": 2}],
                                            [None, None],
                                            [{}, {}],
                                            [{}, {}],
                                            [{}, {}],
                                            [{}, {}]],
                                    subplot_titles=(subtitles))
        fig.append_trace(go.Scatter(
            x=data[0] / str_to_unit[frequency_unit],
            y=calc_corr_group_delay_0 / str_to_unit[group_delay_unit],
            opacity=0.7,
            marker={'color': "red", 'line': {'color': "red"}}
        ), 1, 1)
        fig.append_trace(go.Scatter(
            x=data[0] / str_to_unit[frequency_unit],
            y=calc_corr_group_delay / str_to_unit[group_delay_unit],
            opacity=0.7,
            marker={'color': "green", 'line': {'color': "green"}}
        ), 1, 2)
        for i in range(4):
            fig.append_trace(go.Scatter(
                x=data[0] / str_to_unit[frequency_unit],
                y=data[i * 2 + 1] / str_to_unit[input_units[1]],
                opacity=0.7,
                marker={
                    'color': "blue",
                    'line': {'color': "blue"}
                }), i + 3, 1)
            fig.append_trace(go.Scatter(
                x=data[0] / str_to_unit[frequency_unit],
                y=data[i * 2 + 2] / str_to_unit[input_units[2]],
                opacity=0.7,
                marker={
                    'color': "red",
                    'line': {'color': "red"}
                }), i + 3, 2)

            fig['layout'][f'xaxis{3 + 2*i}'].update(title=xlabels[0] + f' [{frequency_unit}]')
            fig['layout'][f'xaxis{4 + 2*i}'].update(title=xlabels[0] + f' [{frequency_unit}]')
            fig['layout'][f'yaxis{3 + 2*i}'].update(title=ylabels[1] + f' [{input_units[1]}]')
            fig['layout'][f'yaxis{4 + 2*i}'].update(title=ylabels[2] + f' [{input_units[2]}]')

        fig['layout']['xaxis1'].update(title=xlabels[0] + f' [{frequency_unit}]')
        fig['layout']['yaxis1'].update(title=ylabels[0] + f' [{group_delay_unit}]')
        fig['layout']['yaxis2'].update(title=ylabels[0] + f' [{group_delay_unit}]')
        fig['layout']['xaxis2'].update(title=xlabels[0] + f' [{frequency_unit}]')
        fig.update_layout(showlegend=False)
        fig.update_layout(
            autosize=False,
            height=1000)

        plot_info_cont.info(f'you entered {len(data[0])} {xlabels[0]} from {min(data[0]) / str_to_unit[frequency_unit]:.4g}{frequency_unit} to {max(data[0]) / str_to_unit[frequency_unit]:.4g}{frequency_unit}')
        for isn, sname in enumerate(S_names):
            plot_info_cont.info(f'{sname}: {ylabels[1]} {len(data[isn * 2 + 1])} values within the range of {min(data[isn * 2 + 1]) / str_to_unit[input_units[1]]:.4g}{input_units[1]} to {max(data[isn * 2 + 1]) / str_to_unit[input_units[1]]:.4g}{input_units[1]}'
                                f'and {ylabels[2]} {len(data[isn * 2 + 2])} values within the range of {min(data[isn * 2 + 2]) / str_to_unit[input_units[2]]:.4g}{input_units[2]} to {max(data[isn * 2 + 2]) / str_to_unit[input_units[2]]:.4g}{input_units[2]}')

        return fig
    else:
        fig = subplots.make_subplots(rows=1, cols=1)
        return fig

# ANTENNA

def select_antenna_name(antenna_type, container, warning_container):
    selected_antenna_name = ''
    # update the dropdown menu
    antenna_names = det.get_object_names(antenna_type)
    antenna_names.insert(0, f'new {antenna_type}')

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
    antenna_text_input = col2.text_input('', placeholder=f'new unique {antenna_type} name', disabled=disable_new_antenna_name)
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
        if primary and antenna_name in det.get_Antenna_names(page_name):
            det.update_primary(page_name, antenna_name)
        det.set_not_working(page_name, antenna_name, primary)
    else:
        if primary and antenna_name in det.get_Antenna_names(page_name):
            det.update_primary(page_name, antenna_name)
        det.antenna_add_Sparameter(page_name, antenna_name, [s_name], data, primary, protocol, input_units)


# SURFACE CABLE

def select_cable(page_name, main_container, warning_container_top):
    col1_cable, col2_cable, col3_cable = main_container.columns([1,1,1])
    cable_types = []
    cable_stations = []
    cable_channels = []
    if page_name == 'surface_cable':
        cable_types=['Choose an option', '11 meter signal']
        cable_stations = ['Choose an option', 'Station 1 (11 Nanoq)', 'Station 2 (12 Terianniaq)', 'Station 3 (13 Ukaleq)', 'Station 4 (14 Tuttu)', 'Station 5 (15 Umimmak)', 'Station 6 (21 Amaroq)', 'Station 7 (22 Avinngaq)', 'Station 8 (23 Ukaliatsiaq)', 'Station 9 (24 Qappik)','Station 10 (25 Aataaq)']
        cable_channels = ['Choose an option', 'Channel 1 (0)', 'Channel 2 (1)', 'Channel 3 (2)', 'Channel 4 (3)', 'Channel 5 (4)', 'Channel 6 (5)', 'Channel 7 (6)', 'Channel 8 (7)', 'Channel 9 (8)']
    elif page_name == 'downhole_cable':
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

    if cable_name in det.get_object_names(page_name):
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


def insert_cable_to_db(page_name, s_name, cable_name, data_m, data_p, input_units, working, primary, protocol):
    if not working:
        if primary and cable_name in det.get_cable_names(page_name):
            det.update_primary(page_name, cable_name)
        det.set_not_working(page_name, cable_name, primary)
    else:
        if primary and cable_name in det.get_cable_names(page_name):
            det.update_primary(page_name, cable_name)
        det.cable_add_Sparameter(page_name, cable_name, [s_name], data_m, data_p, input_units, primary, protocol)


# IGLU

def select_iglu(page_name, main_container, warning_container):
    col1_I, col2_I, col3_I, col4_I, col5_I = main_container.columns([1,1,1,1,1])

    selected_iglu_name = ''
    iglu_names = det.get_object_names(page_name)
    iglu_names.insert(0, f'new {page_name}')
    drab_names = det.get_object_names('drab_board')

    iglu_dropdown = col1_I.selectbox('Select existing board or enter unique name of new board:', iglu_names)
    if iglu_dropdown == f'new {page_name}':
        disable_new_input = False

        selected_iglu_infos = []
    else:
        disable_new_input = True
        selected_iglu_name = iglu_dropdown
        warning_container.warning(f'You are about to override the {page_name} unit {iglu_dropdown}!')

        # load all the information for this board
        selected_iglu_infos = det.load_board_information(page_name, selected_iglu_name, ['laser_id', 'DRAB_id', 'measurement_temp'])

    new_board_name = col2_I.text_input('', placeholder=f'new unique board name', disabled=disable_new_input)
    if iglu_dropdown == f'new {page_name}':
        selected_iglu_name = new_board_name

    # always editable (maybe want to change the laser serial number)
    if selected_iglu_infos == []:
        laser_serial_name = col3_I.text_input('', placeholder='laser serial', disabled=False)
    else:
        laser_serial_name = col3_I.text_input('', value=selected_iglu_infos[0], disabled=False)
    col3_I.markdown(laser_serial_name)

    # if an exiting IGLU is selected, change the default option to the saved DRAB
    if selected_iglu_infos != []:
        drab_index = drab_names.index(selected_iglu_infos[1])
        drab_names.pop(drab_index)
        drab_names.insert(0, selected_iglu_infos[1])
    else:
        # select golden DRAB as the default option
        golden_drab_index = drab_names.index('Golden_DRAB')
        drab_names.pop(golden_drab_index)
        drab_names.insert(0, f'Golden_DRAB')
    selected_DRAB = col4_I.selectbox('', drab_names)

    temp_list = ['room temp (20°C)', '-50°C', '-40°C', '-30°C', '-20°C', '-10°C', '0°C', '10°C', '30°C', '40°C']
    # if an exiting IGLU is selected, change the default option to the saved temperature
    if selected_iglu_infos != []:
        if selected_iglu_infos[2] == 20:
            saved_temp = 'room temp (20°C)'
        else:
            saved_temp = str(selected_iglu_infos[2]) + '°C'
        temp_index = temp_list.index(saved_temp)
        temp_list.pop(temp_index)
        temp_list.insert(0, saved_temp)
    selected_Temp = col5_I.selectbox('', temp_list)
    if 'room temp' in selected_Temp:
        selected_Temp = int(selected_Temp[len('room temp ('):-3])
    else:
        selected_Temp = int(selected_Temp[:-2])

    return selected_iglu_name, iglu_dropdown, laser_serial_name, selected_DRAB, selected_Temp


def validate_global_iglu(page_name, container_bottom, iglu_name, new_iglu_name, laser_serial_name, channel_working, Sdata_validated, uploaded_data):
    disable_insert_button = True
    name_validation = False
    laser_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if iglu_name == '':
        container_bottom.error('IGLU name is not set')
    elif iglu_name == f'new {page_name}' and (new_iglu_name is None or new_iglu_name == ''):
        container_bottom.error(f'IGLU name dropdown is set to \'new {page_name}\', but no new IGLU name was entered.')
    else:
        name_validation = True

    if laser_serial_name == '' and channel_working:
        container_bottom.error('Laser serial number is not entered.')
    else:
        laser_validation = True

    if name_validation and laser_validation:
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


def insert_iglu_to_db(page_name, s_names, iglu_name, data, input_units, working, primary, protocol, drab_id, laser_id, temp, measurement_time, time_delay):
    if not working:
        if primary and iglu_name in det.get_board_names(page_name):
            # temperature is not used for update_primary (if the board doesn't work, it will not work for every temperature)
            det.update_primary(page_name, iglu_name)
        det.set_not_working(page_name, iglu_name, primary)
    else:
        if primary and iglu_name in det.get_board_names(page_name):
            det.update_primary(page_name, iglu_name, temp)
        det.iglu_add_Sparameters(page_name, s_names, iglu_name, drab_id, laser_id, temp, data, measurement_time, primary, time_delay, protocol, input_units)


# DRAB

def select_drab(page_name, main_container, warning_container):
    col1_I, col2_I, col3_I, col4_I, col5_I, col6_I = main_container.columns([1.2,1,1,0.8,1,1])

    selected_drab_name = ''
    drab_names = det.get_object_names(page_name)
    drab_names.insert(0, f'new {page_name}')
    iglu_names = det.get_object_names('iglu_board')

    drab_dropdown = col1_I.selectbox('Select existing board or enter unique name of new board:', drab_names)
    if drab_dropdown == f'new {page_name}':
        disable_new_input = False

        selected_drab_infos = []
    else:
        disable_new_input = True
        selected_drab_name = drab_dropdown
        warning_container.warning(f'You are about to override the {page_name} unit {drab_dropdown}!')

        # load all the information for this board
        selected_drab_infos = det.load_board_information(page_name, selected_drab_name, ['photodiode_serial', 'channel_id', 'IGLU_id', 'measurement_temp'])

    new_board_name = col2_I.text_input('', placeholder=f'new unique board name', disabled=disable_new_input)
    if drab_dropdown == f'new {page_name}':
        selected_drab_name = new_board_name

    # always editable (maybe want to change the photodiode serial number)
    if selected_drab_infos == []:
        photodiode_number = col3_I.text_input('', placeholder='photodiode serial', disabled=False)
    else:
        photodiode_number = col3_I.text_input('', value=selected_drab_infos[0], disabled=False)
    col3_I.markdown(photodiode_number)

    channel_numbers = ['Choose a channel-id', '0', '1', '2', '3']
    # if an exiting drab is selected, change the default option to the saved IGLU
    if selected_drab_infos != []:
        cha_index = channel_numbers.index(str(selected_drab_infos[1]))
        channel_numbers.pop(cha_index)
        channel_numbers.insert(0, str(selected_drab_infos[1]))
    selected_channel_id = col4_I.selectbox('', channel_numbers)

    # if an exiting drab is selected, change the default option to the saved IGLU
    if selected_drab_infos != []:
        iglu_index = iglu_names.index(selected_drab_infos[2])
        iglu_names.pop(iglu_index)
        iglu_names.insert(0, selected_drab_infos[2])
    else:
        # select golden IGLU as the default option
        golden_iglu_index = iglu_names.index('Golden_IGLU')
        iglu_names.pop(golden_iglu_index)
        iglu_names.insert(0, f'Golden_IGLU')
    selected_IGLU = col5_I.selectbox('', iglu_names)

    temp_list = ['room temp (20°C)', '-50°C', '-40°C', '-30°C', '-20°C', '-10°C', '0°C', '10°C', '30°C', '40°C']
    # if an exiting DRAB is selected, change the default option to the saved temperature
    if selected_drab_infos != []:
        if selected_drab_infos[3] == 20:
            saved_temp = 'room temp (20°C)'
        else:
            saved_temp = str(selected_drab_infos[3]) + '°C'
        temp_index = temp_list.index(saved_temp)
        temp_list.pop(temp_index)
        temp_list.insert(0, saved_temp)
    selected_Temp = col6_I.selectbox('', temp_list)
    if 'room temp' in selected_Temp:
        selected_Temp = int(selected_Temp[len('room temp ('):-3])
    else:
        selected_Temp = int(selected_Temp[:-2])

    return selected_drab_name, drab_dropdown, photodiode_number, selected_channel_id,selected_IGLU, selected_Temp


def validate_global_drab(page_name, container_bottom, drab_name, new_drab_name, photodiode_number, channel_id, channel_working, Sdata_validated, uploaded_data):
    disable_insert_button = True
    name_validation = False
    input_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if drab_name == '':
        container_bottom.error('DRAB name is not set')
    elif drab_name == f'new {page_name}' and (new_drab_name is None or new_drab_name == ''):
        container_bottom.error(f'DRAB name dropdown is set to \'new {page_name}\', but no new DRAB name was entered.')
    else:
        name_validation = True

    if (photodiode_number == '' and channel_working) or ('Choose' in channel_id and channel_working):
        container_bottom.error('Not all input options are entered.')
    else:
        input_validation = True

    if name_validation and input_validation:
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


def insert_drab_to_db(page_name, s_names, drab_name, data, input_units, working, primary, protocol, iglu_id, photodiode_id, channel_id, temp, measurement_time, time_delay):
    if not working:
        if primary and drab_name in det.get_board_names(page_name):
            # temperature is not used for update_primary (if the board doesn't work, it will not work for every temperature)
            det.update_primary(page_name, drab_name, channel_id=int(channel_id))
        det.set_not_working(page_name, drab_name, primary, int(channel_id))
    else:
        if primary and drab_name in det.get_board_names(page_name):
            det.update_primary(page_name, drab_name, temp, int(channel_id))
        det.drab_add_Sparameters(page_name, s_names, drab_name, iglu_id, photodiode_id, int(channel_id), temp, data, measurement_time, primary, time_delay, protocol, input_units)

# SURFACE

def select_surface(page_name, main_container, warning_container):
    col1_I, col2_I, col3_I, col4_I = main_container.columns([1,1,1,1])

    selected_surface_name = ''
    surface_names = det.get_object_names(page_name)
    surface_names.insert(0, f'new {page_name}')

    surface_dropdown = col1_I.selectbox('Select existing board or enter unique name of new board:', surface_names)
    if surface_dropdown == f'new {page_name}':
        disable_new_input = False

        selected_surface_infos = []
    else:
        disable_new_input = True
        selected_surface_name = surface_dropdown
        warning_container.warning(f'You are about to override the {page_name} unit {surface_dropdown}!')

        # load all the information for this board
        selected_surface_infos = det.load_board_information(page_name, selected_surface_name, ['channel_id', 'measurement_temp'])

    new_board_name = col2_I.text_input('', placeholder=f'new unique board name', disabled=disable_new_input)
    if surface_dropdown == f'new {page_name}':
        selected_surface_name = new_board_name

    channel_numbers = ['Choose a channel-id', '0', '1', '2', '3', '4']
    # if an exiting drab is selected, change the default option to the saved IGLU
    if selected_surface_infos != []:
        cha_index = channel_numbers.index(str(selected_surface_infos[0]))
        channel_numbers.pop(cha_index)
        channel_numbers.insert(0, str(selected_surface_infos[0]))
    selected_channel_id = col3_I.selectbox('', channel_numbers)

    temp_list = ['room temp (20°C)', '-50°C', '-40°C', '-30°C', '-20°C', '-10°C', '0°C', '10°C', '30°C', '40°C']
    # if an exiting DRAB is selected, change the default option to the saved temperature
    if selected_surface_infos != []:
        if selected_surface_infos[1] == 20:
            saved_temp = 'room temp (20°C)'
        else:
            saved_temp = str(selected_surface_infos[1]) + '°C'
        temp_index = temp_list.index(saved_temp)
        temp_list.pop(temp_index)
        temp_list.insert(0, saved_temp)
    selected_Temp = col4_I.selectbox('', temp_list)
    if 'room temp' in selected_Temp:
        selected_Temp = int(selected_Temp[len('room temp ('):-3])
    else:
        selected_Temp = int(selected_Temp[:-2])

    return selected_surface_name, surface_dropdown, selected_channel_id, selected_Temp


def validate_global_surface(page_name, container_bottom, surface_name, new_surface_name, channel_id, channel_working, Sdata_validated, uploaded_data):
    disable_insert_button = True
    name_validation = False
    input_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if surface_name == '':
        container_bottom.error(f'{page_name} name is not set')
    elif surface_name == f'new {page_name}' and (new_surface_name is None or new_surface_name == ''):
        container_bottom.error(f'{page_name} name dropdown is set to \'new {page_name}\', but no new {page_name} name was entered.')
    else:
        name_validation = True

    if 'Choose' in channel_id and channel_working:
        container_bottom.error('Not all input options are entered.')
    else:
        input_validation = True

    if name_validation and input_validation:
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


def insert_surface_to_db(page_name, s_names, surface_name, data, input_units, working, primary, protocol, channel_id, temp, measurement_time, time_delay):
    if not working:
        if primary and surface_name in det.get_board_names(page_name):
            det.update_primary(page_name, surface_name, channel_id=int(channel_id))
        det.set_not_working(page_name, surface_name, primary, int(channel_id))
    else:
        if primary and surface_name in det.get_board_names(page_name):
            det.update_primary(page_name, surface_name, temp, int(channel_id))
        det.surface_add_Sparameters(page_name, s_names, surface_name, int(channel_id), temp, data, measurement_time, primary, time_delay, protocol, input_units)

# downhole

def select_downhole(page_name, main_container, warning_container):
    col1_I, col2_I, col3_I, col4_I, col5_I, col6_I, col7_I = main_container.columns([1.35,1.1,0.9,0.9,1,1,0.75])

    selected_downhole_name = ''
    downhole_names = det.get_object_names(page_name)
    downhole_names.insert(0, f'new fiber')

    downhole_dropdown = col1_I.selectbox('Select existing fiber or enter unique name of new fiber:', downhole_names)
    if downhole_dropdown == f'new fiber':
        disable_new_input = False

        selected_downhole_infos = []
    else:
        disable_new_input = True
        selected_downhole_name = downhole_dropdown
        warning_container.warning(f'You are about to override the fiber unit {downhole_dropdown}!')

        # load all the information for this board
        selected_downhole_infos = det.load_board_information(page_name, selected_downhole_name, ['breakout', 'breakout_channel', 'IGLU_id', 'DRAB_id', 'measurement_temp'])

    new_board_name = col2_I.text_input('', placeholder=f'new unique fiber name', disabled=disable_new_input)
    if downhole_dropdown == f'new fiber':
        selected_downhole_name = new_board_name

    breakout_ids = ['breakout-id', '1', '2', '3']
    # if an exiting fiber is selected, change the default option to the saved number
    if selected_downhole_infos != []:
        breakout_index = breakout_ids.index(str(selected_downhole_infos[0]))
        breakout_ids.pop(breakout_index)
        breakout_ids.insert(0, str(selected_downhole_infos[0]))
    selected_breakout_id = col3_I.selectbox('', breakout_ids)

    breakout_cha_ids = ['breakout channel-id', 'p1', 'p2', 'p3', 's1', 's2', 's3']
    # if an exiting fiber is selected, change the default option to the saved number
    if selected_downhole_infos != []:
        breakout_cha_index = breakout_cha_ids.index(str(selected_downhole_infos[1]))
        breakout_cha_ids.pop(breakout_cha_index)
        breakout_cha_ids.insert(0, str(selected_downhole_infos[1]))
    selected_breakout_cha_id = col4_I.selectbox('', breakout_cha_ids)

    # if an exiting fiber is selected, change the default option to the saved IGLU
    iglu_names = det.get_object_names('iglu_board')
    if selected_downhole_infos != []:
        iglu_index = iglu_names.index(selected_downhole_infos[2])
        iglu_names.pop(iglu_index)
        iglu_names.insert(0, selected_downhole_infos[2])
    else:
        # select golden IGLU as the default option
        golden_iglu_index = iglu_names.index('Golden_IGLU')
        iglu_names.pop(golden_iglu_index)
        iglu_names.insert(0, f'Golden_IGLU')
    selected_IGLU = col5_I.selectbox('', iglu_names)

    # if an exiting fiber is selected, change the default option to the saved DRAB
    drab_names = det.get_object_names('drab_board')
    if selected_downhole_infos != []:
        drab_index = drab_names.index(selected_downhole_infos[3])
        drab_names.pop(drab_index)
        drab_names.insert(0, selected_downhole_infos[3])
    else:
        # select golden DRAB as the default option
        golden_drab_index = drab_names.index('Golden_DRAB')
        drab_names.pop(golden_drab_index)
        drab_names.insert(0, f'Golden_DRAB')
    selected_DRAB = col6_I.selectbox('', drab_names)

    temp_list = ['room temp (20°C)', '-50°C', '-40°C', '-30°C', '-20°C', '-10°C', '0°C', '10°C', '30°C', '40°C']
    # if an exiting fiber is selected, change the default option to the saved temperature
    if selected_downhole_infos != []:
        if selected_downhole_infos[4] == 20:
            saved_temp = 'room temp (20°C)'
        else:
            saved_temp = str(selected_downhole_infos[4]) + '°C'
        temp_index = temp_list.index(saved_temp)
        temp_list.pop(temp_index)
        temp_list.insert(0, saved_temp)
    selected_Temp = col7_I.selectbox('', temp_list)
    if 'room temp' in selected_Temp:
        selected_Temp = int(selected_Temp[len('room temp ('):-3])
    else:
        selected_Temp = int(selected_Temp[:-2])

    return selected_downhole_name, downhole_dropdown, selected_breakout_id, selected_breakout_cha_id, selected_IGLU, selected_DRAB, selected_Temp


def validate_global_downhole(page_name, container_bottom, surface_name, new_surface_name, breakout_id, breakout_cha_id, channel_working, Sdata_validated, uploaded_data):
    disable_insert_button = True
    name_validation = False
    input_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if surface_name == '':
        container_bottom.error(f'fiber name is not set')
    elif surface_name == f'new fiber' and (new_surface_name is None or new_surface_name == ''):
        container_bottom.error(f'fiber name dropdown is set to \'new fiber\', but no new fiber name was entered.')
    else:
        name_validation = True

    if ('breakout' in breakout_id and channel_working) or ('breakout' in breakout_cha_id and channel_working):
        container_bottom.error('Not all input options are entered.')
    else:
        input_validation = True

    if name_validation and input_validation:
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


def insert_downhole_to_db(page_name, s_names, downhole_name, data, input_units, working, primary, protocol, breakout_id, breakout_cha_id, iglu_id, drab_id, temp, measurement_time, time_delay):
    if not working:
        if primary and downhole_name in det.get_board_names(page_name):
            det.update_primary(page_name, downhole_name)
        det.set_not_working(page_name, downhole_name, primary)
    else:
        if primary and downhole_name in det.get_board_names(page_name):
            det.update_primary(page_name, downhole_name, temp)
        det.downhole_add_Sparameters(page_name, s_names, downhole_name, int(breakout_id), breakout_cha_id, iglu_id, drab_id, temp, data, measurement_time, primary, time_delay, protocol, input_units)
