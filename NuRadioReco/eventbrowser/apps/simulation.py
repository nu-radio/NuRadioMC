from __future__ import absolute_import, division, print_function  # , unicode_literals
from dash.dependencies import Input, Output, State
import dash
import radiotools.helper as hp
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import plotly
from plotly import tools
import json
from app import app
import dataprovider
from NuRadioReco.utilities import units
from NuRadioReco.utilities import templates
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
import numpy as np
import logging
logger = logging.getLogger('traces')

provider = dataprovider.DataProvider()
template_provider = templates.Templates()

layout = html.Div([
    #Sim Traces Plot
    html.Div([
        html.Div([
            html.Div('Sim Traces', className='panel-heading'),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div('Signal Type', className=''),
                            dcc.Checklist(
                                id='sim-traces-signal-types',
                                options = [
                                    {'label': 'direct', 'value': 'direct'},
                                    {'label': 'reflected/refracted', 'value': 'indirect'}
                                ],
                                values = ['direct'],
                                className='sim-trace-option'
                            )
                        ], className=''),
                        html.Div([
                            html.Div('Polarization', className=''),
                            dcc.RadioItems(
                                id='sim-traces-polarization',
                                options = [
                                    {'label': '0', 'value': 0},
                                    {'label': '1', 'value': 1},
                                    {'label': '2', 'value': 2}
                                ],
                                value = 0,
                                className=''
                            )
                        ], className='sim-trace-option')
                    ], className='sim-trace-options'),
                    html.Div([
                        dcc.Graph(id='sim-traces')
                    ]),
                ], className='panel-body', style={'min-height': '500px'})
            ], className='panel panel-default mb-2', style={'flex': '1'}),
            #Sim Spectrum Plot
            html.Div([
                html.Div('Sim Spectrum', className='panel-heading'),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div('Signal Type', className=''),
                            dcc.Checklist(
                                id='sim-spectrum-signal-types',
                                options = [
                                    {'label': 'direct', 'value': 'direct'},
                                    {'label': 'reflected/refracted', 'value': 'indirect'}
                                ],
                                values = ['direct'],
                                className='sim-trace-option'
                            )
                        ], className=''),
                        html.Div([
                            html.Div('Polarization', className=''),
                            dcc.RadioItems(
                                id='sim-spectrum-polarization',
                                options = [
                                    {'label': '0', 'value': 0},
                                    {'label': '1', 'value': 1},
                                    {'label': '2', 'value': 2}
                                ],
                                value = 0,
                                className=''
                            )
                        ], className='sim-trace-option')
                    ], className='sim-trace-options'),
                    html.Div([
                        dcc.Graph(id='sim-spectrum')
                    ]),
                ], className='panel-body', style={'min-height': '500px'})
            ], className='panel panel-default mb-2', style={'flex': '1'})
        ], style={'display': 'flex'}),
        html.Div([
            html.Div('Simulated Event', className='panel-heading'),
            html.Div([
                dcc.Graph(id='sim-event-3d')
            ],
            className='panel-body'
            )
        ],
        className='panel panel-default',
        style = {'width': '50%'}
        )
    ])

@app.callback(
    Output('sim-traces', 'figure'),
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value'),
    Input('sim-traces-signal-types', 'values'),
    Input('sim-traces-polarization', 'value')],
    [State('user_id', 'children'),
     State('station_id', 'children')]
)
def update_sim_trace_plot(i_event, filename, signal_types, polarization, juser_id, jstation_id):
    if filename is None:
        return {}
    user_id = json.loads(juser_id)
    station_id = json.loads(jstation_id)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_stations()[0]
    sim_station = station.get_sim_station()
    if sim_station is None:
        return {}
    fig = tools.make_subplots(rows=1, cols=1)
    for i_channel, channel in enumerate(sim_station.get_channels()):
        if 'direct' in signal_types:
            fig.append_trace(
                go.Scatter(
                    x=channel[0].get_times()/units.ns,
                    y=channel[0].get_trace()[polarization]/units.mV,
                    opacity=0.7,
                    line = {
                        'color': colors[i_channel % len(colors)]
                    },
                    name = 'Channel {}'.format(i_channel),
                    legendgroup=str(i_channel)
                ), 1, 1
            )
        if 'indirect' in signal_types:
            if 'direct' in signal_types:
                name = ''
            else:
                name = 'Channel {}'.format(i_channel)
            fig.append_trace(
                go.Scatter(
                    x=channel[1].get_times()/units.ns,
                    y=channel[1].get_trace()[polarization]/units.mV,
                    opacity=0.7,
                    legendgroup=str(i_channel),
                    name = name,
                    line = {
                        'dash': 'dot',
                        'color': colors[i_channel % len(colors)]
                        }
                ), 1, 1
            )
    fig['layout'].update(
        legend = {
            'orientation': 'h',
            'y': 1.2
            },
        xaxis={'title': 't [ns]'},
        yaxis={'title': 'voltage [mV]'}
    )
    return fig

@app.callback(
    Output('sim-spectrum', 'figure'),
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value'),
    Input('sim-spectrum-signal-types', 'values'),
    Input('sim-spectrum-polarization', 'value')],
    [State('user_id', 'children'),
     State('station_id', 'children')]
)
def update_sim_spectrum_plot(i_event, filename, signal_types, polarization, juser_id, jstation_id):
    if filename is None:
        return {}
    user_id = json.loads(juser_id)
    station_id = json.loads(jstation_id)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_stations()[0]
    sim_station = station.get_sim_station()
    if sim_station is None:
        return {}
    fig = tools.make_subplots(rows=1, cols=1)
    for i_channel, channel in enumerate(sim_station.get_channels()):
        if 'direct' in signal_types:
            fig.append_trace(
                go.Scatter(
                    x=channel[0].get_frequencies()/units.MHz,
                    y=np.abs(channel[0].get_frequency_spectrum()[polarization])/units.mV,
                    opacity=0.7,
                    line = {
                        'color': colors[i_channel % len(colors)]
                    },
                    name = 'Channel {}'.format(i_channel),
                    legendgroup=str(i_channel)
                ), 1, 1
            )
        if 'indirect' in signal_types:
            if 'direct' in signal_types:
                name = ''
            else:
                name = 'Channel {}'.format(i_channel)
            if len(channel) >1:
                freqs = channel[1].get_frequencies()/units.MHz
                spec = np.abs(channel[1].get_frequency_spectrum()/units.mV)
            else:
                freqs = channel[0].get_frequencies()/units.MHz
                spec = np.zeros(len(freqs))
            fig.append_trace(
                go.Scatter(
                    x=freqs,
                    y=spec,
                    opacity=0.7,
                    legendgroup=str(i_channel),
                    name = name,
                    line = {
                        'dash': 'dot',
                        'color': colors[i_channel % len(colors)]
                        }
                ), 1, 1
            )
    fig['layout'].update(
        legend = {
            'orientation': 'h',
            'y': 1.2
            },
        xaxis={'title': 'f [MHz]'},
        yaxis={'title': 'voltage [mV]'}
    )
    return fig
    
@app.callback(
    Output('sim-event-3d', 'figure'),
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value')],
    [State('user_id', 'children'),
     State('station_id', 'children')]
     )
def update_sim_event_3d(i_event, filename, juser_id, jstation_id):
    if filename is None:
        return {}
    user_id = json.loads(juser_id)
    station_id = json.loads(jstation_id)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_stations()[0]
    sim_station = station.get_sim_station()
    if sim_station is None:
        return {}
    vertex = sim_station.get_parameter(stnp.nu_vertex)
    neutrino_path = hp.spherical_to_cartesian(sim_station.get_parameter(stnp.nu_zenith), sim_station.get_parameter(stnp.nu_azimuth))
    fig = go.Figure(
        data = [
            go.Scatter3d(
                x = [vertex[0]],
                y = [vertex[1]],
                z = [vertex[2]],
                mode = 'markers',
                name = 'Interaction Vertex'
                ),
            go.Scatter3d(
                x = [0],
                y = [0],
                z = [0],
                mode = 'markers',
                name = 'Station'
                ),
            go.Scatter3d(
                x = [vertex[0], vertex[0] + 500*neutrino_path[0]],
                y = [vertex[1], vertex[1] + 500*neutrino_path[1]],
                z = [vertex[2], vertex[2] + 500*neutrino_path[2]],
                name = 'Neutrino Direction',
                mode = 'lines'
                )
            ],
            layout = go.Layout(
                width = 500,
                height = 500,
                legend = {
                    'orientation': 'h',
                    'y': 1.1
                },
                scene = {
                'aspectmode': 'manual',
                'aspectratio': {
                    'x': 2,
                    'y': 2,
                    'z': 1
                },
                'xaxis' : {
                    'range': [-1000,1000]
                },
                'yaxis' : {
                    'range': [-1000, 1000]
                },
                'zaxis' : {
                    'range': [-1000,0]
                }
                }
            )
        )
    return fig

    