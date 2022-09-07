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
