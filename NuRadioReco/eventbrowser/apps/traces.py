from __future__ import absolute_import, division, print_function  # , unicode_literals
from dash.dependencies import Input, Output, State
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import plotly
from plotly import tools
from plotly import subplots
import json
from app import app
import dataprovider
from NuRadioReco.utilities import units
from NuRadioReco.utilities import templates
from NuRadioReco.utilities import trace_utilities
from NuRadioReco.utilities import fft
from NuRadioReco.utilities import geometryUtilities
from NuRadioReco.detector import detector
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.eventbrowser.default_layout import default_layout

import NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_trace
import NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_spectrum
import NuRadioReco.eventbrowser.apps.trace_plots.channel_time_trace
import NuRadioReco.eventbrowser.apps.trace_plots.channel_spectrum
import NuRadioReco.eventbrowser.apps.trace_plots.multi_channel_plot

import NuRadioReco.detector.antennapattern
import numpy as np
import logging
import os

logger = logging.getLogger('traces')

provider = dataprovider.DataProvider()
# if environment variable for templates is set, we use it, otherwise we later
# get the template directory from user input
if 'NURADIORECOTEMPLATES' in os.environ:
    template_provider = templates.Templates(os.environ.get('NURADIORECOTEMPLATES'))
else:
    template_provider = templates.Templates('')
det = detector.Detector()
antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()

layout = html.Div([
    html.Div(id='trigger-trace', style={'display': 'none'}),
    html.Div([
        html.Div([
            html.Div('Electric Field Traces', className='panel-heading'),
            html.Div([
            dcc.Graph(id='efield-trace')
            ], className='panel-body')
        ], className='panel panel-default', style={'flex': '1'}),
        html.Div([
            html.Div('Electric Field Spectrum', className='panel-heading'),
            html.Div([
            dcc.Graph(id='efield-spectrum')
            ], className='panel-body')
        ], className='panel panel-default', style={'flex': '1'})
    ], style={'display': 'flex'}),
    html.Div([
        html.Div([
            html.Div('Channel Traces', className='panel-heading'),
            html.Div([
                dcc.Graph(id='time-trace')
            ], className='panel-body')
        ], className='panel panel-default', style={'flex': '1'}),
        html.Div([
            html.Div('Channel Spectrum', className='panel-heading'),
            html.Div([
                dcc.Graph(id='channel-spectrum')
            ], className='panel-body')
        ], className='panel panel-default', style={'flex': '1'})
    ], style={'display': 'flex'}),
    html.Div([
        html.Div([
            html.Div('Individual Channels', className='panel-heading'),
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Dropdown(id='dropdown-traces',
                            options=[
                                {'label': 'calibrated trace', 'value': 'trace'},
                                {'label': 'cosmic-ray template', 'value': 'crtemplate'},
                                {'label': 'neutrino template', 'value': 'nutemplate'},
                                {'label': 'envelope', 'value': 'envelope'},
                                {'label': 'from rec. E-field', 'value': 'efield'}
                            ],
                            multi=True,
                            value=["trace"]
                        )
                    ], style={'flex': '1'}),
                    html.Div([
                        dcc.Dropdown(id='dropdown-trace-info',
                            options=[
                                {'label': 'RMS', 'value': 'RMS'},
                                {'label': 'L1', 'value': 'L1'}
                            ],
                            multi=True,
                            value=["RMS", "L1"]
                        )
                    ], style={'flex': '1'}),
                ], style={'display': 'flex'}),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div('Template directory', className='input-group-addon'),
                            dcc.Input(id='template-directory-input', placeholder='template directory', className='form-control'),
                            html.Div([
                                html.Button('load', id='open-template-button', className='btn btn-default')
                            ], className='input-group-btn')
                        ], className='input-group', id='template-input-group')
                    ], style={'flex': '1'}),
                    html.Div('', style={'flex': '1'})
                ], style={'display': 'flex'}),
                dcc.Graph(id='time-traces')
            ], className='panel-body')
        ], className='panel panel-default', style={'flex': '1'})
    ], style={'display': 'flex'})
])


@app.callback(
    dash.dependencies.Output('efield-trace', 'figure'),
    [dash.dependencies.Input('trigger-trace', 'children'),
     dash.dependencies.Input('event-counter-slider', 'value'),
     dash.dependencies.Input('filename', 'value'),
     dash.dependencies.Input('station-id-dropdown', 'value')],
     [State('user_id', 'children')])
def update_time_efieldtrace(trigger, evt_counter, filename, station_id, juser_id):
    return NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_trace.update_time_efieldtrace(trigger, evt_counter, filename, station_id, juser_id, provider)

@app.callback(
    dash.dependencies.Output('efield-spectrum', 'figure'),
    [dash.dependencies.Input('trigger-trace', 'children'),
     dash.dependencies.Input('event-counter-slider', 'value'),
     dash.dependencies.Input('filename', 'value'),
     dash.dependencies.Input('station-id-dropdown', 'value')],
     [State('user_id', 'children')])
def update_efield_spectrum(trigger, evt_counter, filename, station_id, juser_id):
    return NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_spectrum.update_efield_spectrum(trigger, evt_counter, filename, station_id, juser_id, provider)

@app.callback(
    dash.dependencies.Output('time-trace', 'figure'),
    [dash.dependencies.Input('trigger-trace', 'children'),
     dash.dependencies.Input('event-counter-slider', 'value'),
     dash.dependencies.Input('filename', 'value'),
     dash.dependencies.Input('station-id-dropdown', 'value')],
     [State('user_id', 'children')])
def update_time_trace(trigger, evt_counter, filename, station_id, juser_id):
    return NuRadioReco.eventbrowser.apps.trace_plots.channel_time_trace.update_time_trace(trigger, evt_counter, filename, station_id, juser_id, provider)

@app.callback(
    dash.dependencies.Output('channel-spectrum', 'figure'),
    [dash.dependencies.Input('trigger-trace', 'children'),
     dash.dependencies.Input('event-counter-slider', 'value'),
     dash.dependencies.Input('filename', 'value'),
     dash.dependencies.Input('station-id-dropdown', 'value')],
     [State('user_id', 'children')])
def update_channel_spectrum(trigger, evt_counter, filename, station_id, juser_id):
    return NuRadioReco.eventbrowser.apps.trace_plots.channel_spectrum.update_channel_spectrum(trigger, evt_counter, filename, station_id, juser_id, provider)

@app.callback(
    dash.dependencies.Output('dropdown-traces', 'options'),
    [dash.dependencies.Input('event-counter-slider', 'value'),
     dash.dependencies.Input('filename', 'value'),
     dash.dependencies.Input('station-id-dropdown', 'value')],
     [State('user_id', 'children')]
)
def get_dropdown_traces_options(evt_counter, filename, station_id, juser_id):
    if filename is None or station_id is None:
        return []
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    options=[
        {'label': 'calibrated trace', 'value': 'trace'},
        {'label': 'cosmic-ray template', 'value': 'crtemplate'},
        {'label': 'neutrino template', 'value': 'nutemplate'},
        {'label': 'envelope', 'value': 'envelope'},
        {'label': 'from rec. E-field', 'value': 'recefield'}
    ]
    if station.get_sim_station() is not None:
        if len(station.get_sim_station().get_electric_fields()) > 0:
            options.append({'label': 'from sim. E-field', 'value': 'simefield'})
    return options

@app.callback(
    dash.dependencies.Output('time-traces', 'figure'),
    [dash.dependencies.Input('event-counter-slider', 'value'),
     dash.dependencies.Input('filename', 'value'),
     dash.dependencies.Input('dropdown-traces', 'value'),
     dash.dependencies.Input('dropdown-trace-info', 'value'),
     dash.dependencies.Input('station-id-dropdown', 'value'),
     dash.dependencies.Input('open-template-button', 'n_clicks_timestamp')],
     [State('user_id', 'children'),
     State('template-directory-input', 'value')])
def update_multi_channel_plot(evt_counter, filename, dropdown_traces, dropdown_info, station_id, open_template_timestamp, juser_id, template_directory):
    return NuRadioReco.eventbrowser.apps.trace_plots.multi_channel_plot.update_multi_channel_plot(evt_counter, filename, dropdown_traces, dropdown_info, station_id, open_template_timestamp, juser_id, template_directory, provider)



@app.callback(
    Output('template-input-group', 'style'),
    [Input('dropdown-traces', 'value')]
)
def show_template_input(trace_dropdown_options):
    if 'NURADIORECOTEMPLATES' in os.environ:
        return {'display': 'none'}
    if 'crtemplate' in trace_dropdown_options or 'nutemplate' in trace_dropdown_options:
        return {}
    return {'display': 'none'}
