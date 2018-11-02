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

polarizaiton_names = ['E_r', 'E_theta', 'E_phi']
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
                            html.Div('Event Type'),
                            dcc.RadioItems(
                                id='sim-traces-event-types',
                                options = [
                                    {'label': 'Neutrino', 'value': 'nu'},
                                    {'label': 'Cosmic Ray', 'value': 'cr'}
                                ],
                                value = 'nu',
                                className='sim-trace-option'
                            )
                        ], className=''),
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
                        ], className='', id='sim-traces-signal-types-container')
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
                            html.Div('Event Type'),
                            dcc.RadioItems(
                                id='sim-spectrum-event-types',
                                options = [
                                    {'label': 'Neutrino', 'value': 'nu'},
                                    {'label': 'Cosmic Ray', 'value': 'cr'}
                                ],
                                value = 'nu',
                                className='sim-trace-option'
                            )
                        ], className=''),
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
                        ], className='', id='sim-spectrum-signal-types-container')
                    ], className='sim-trace-options'),
                    html.Div([
                        dcc.Graph(id='sim-spectrum')
                    ]),
                ], className='panel-body', style={'min-height': '500px'})
            ], className='panel panel-default mb-2', style={'flex': '1'})
        ], style={'display': 'flex'}),
        html.Div([
            html.Div([
                html.Div('Simulated Event', className='panel-heading'),
                html.Div([
                    dcc.Graph(id='sim-event-3d', style={'flex': '1'}),
                    html.Div([
                        dcc.Dropdown(id='sim-station-properties-dropdown', options=[], multi=True, value=[]),
                        html.Div(id='sim-station-properties-table', className='table table-striped')
                    ], style={'flex': '1', 'min-height': '500px'})
                ], className='panel-body', style={'display': 'flex'})
            ], className='panel panel-default', style = {'flex': 'none', 'width': '50%'}),
        ], style={'display': 'flex'})
    ])
    
@app.callback(
    Output('sim-traces-signal-types-container', 'style'),
    [Input('sim-traces-event-types', 'value')]
)
def show_signal_traces_signal_types(event_type):
    if event_type == 'nu':
        return {}
    else:
        return {'display': 'none'}

@app.callback(
    Output('sim-spectrum-signal-types-container', 'style'),
    [Input('sim-spectrum-event-types', 'value')]
)
def show_signal_spectrum_signal_types(event_type):
    if event_type == 'nu':
        return {}
    else:
        return {'display': 'none'}

@app.callback(
    Output('sim-traces', 'figure'),
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value'),
    Input('sim-traces-event-types', 'value'),
    Input('sim-traces-signal-types', 'values')],
    [State('user_id', 'children'),
     State('station_id', 'children')]
)
def update_sim_trace_plot(i_event, filename, event_type, signal_types, juser_id, jstation_id):
    if filename is None:
        return {}
    user_id = json.loads(juser_id)
    station_id = json.loads(jstation_id)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_stations()[0]
    sim_station = station.get_sim_station()
    visibility_settings = ['legendonly', True, True]
    linestyles = ['dot', 'dash', 'solid']
    if sim_station is None:
        return {}
    fig = tools.make_subplots(rows=1, cols=1)
    try:
        if event_type == 'nu':
            for i_channel, channel in enumerate(sim_station.iter_channels()):
                if 'direct' in signal_types:
                    for polarization in range(0,3):
                        fig.append_trace(
                            go.Scatter(
                                x=channel[0].get_times()/units.ns,
                                y=channel[0].get_trace()[polarization]/units.mV,
                                opacity=1.,
                                line = {
                                    'color': colors[i_channel % len(colors)],
                                    'dash': linestyles[polarization]
                                },
                                name = 'Channel {} ({})'.format(i_channel, polarizaiton_names[polarization]),
                                legendgroup=str(i_channel),
                                visible=visibility_settings[polarization]
                            ), 1, 1
                        )
                if 'indirect' in signal_types:
                    for polarization in range(0,3):
                        if 'direct' in signal_types:
                            name = ''
                        else:
                            name = 'Channel {} ({})'.format(i_channel, polarizaiton_names[polarization])
                        fig.append_trace(
                            go.Scatter(
                                x=channel[1].get_times()/units.ns,
                                y=channel[1].get_trace()[polarization]/units.mV,
                                opacity=0.5,
                                legendgroup=str(i_channel),
                                name = name,
                                line = {
                                    'dash': linestyles[polarization],
                                    'color': colors[i_channel % len(colors)]
                                    },
                                visible=visibility_settings[polarization]
                            ), 1, 1
                        )
        else:
            for polarization in range(0,3):
                fig.append_trace(
                    go.Scatter(
                        x=sim_station.get_times()/units.ns,
                        y=sim_station.get_trace()[polarization]/units.mV,
                        opacity=0.7,
                        line = {
                            'color': colors[polarization % len(colors)],
                            'dash': linestyles[polarization]
                        },
                        name = str(polarizaiton_names[polarization]),
                        visible=visibility_settings[polarization]
                    ), 1, 1
                )
    except:
        return {}
    fig['layout'].update(
        xaxis={'title': 't [ns]'},
        yaxis={'title': 'voltage [mV]'}
    )
    return fig

@app.callback(
    Output('sim-spectrum', 'figure'),
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value'),
    Input('sim-spectrum-event-types', 'value'),
    Input('sim-spectrum-signal-types', 'values')],
    [State('user_id', 'children'),
     State('station_id', 'children')]
)
def update_sim_spectrum_plot(i_event, filename, event_type, signal_types, juser_id, jstation_id):
    if filename is None:
        return {}
    user_id = json.loads(juser_id)
    station_id = json.loads(jstation_id)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_stations()[0]
    sim_station = station.get_sim_station()
    visibility_settings = ['legendonly', True, True]
    linestyles = ['dot', 'dash', 'solid']
    if sim_station is None:
        return {}
    fig = tools.make_subplots(rows=1, cols=1)
    try:
        if event_type == 'nu':
            for i_channel, channel in enumerate(sim_station.iter_channels()):
                if 'direct' in signal_types:
                    for polarization in range(0,3):
                        fig.append_trace(
                            go.Scatter(
                                x=channel[0].get_frequencies()/units.MHz,
                                y=np.abs(channel[0].get_frequency_spectrum()[polarization])/units.mV,
                                opacity=1.,
                                line = {
                                    'color': colors[i_channel % len(colors)],
                                    'dash': linestyles[polarization]
                                },
                                name = 'Channel {} ({})'.format(i_channel, polarizaiton_names[polarization]),
                                legendgroup=str(i_channel),
                                visible=visibility_settings[polarization]
                            ), 1, 1
                        )
                if 'indirect' in signal_types:
                    if len(channel) >1:
                        freqs = channel[1].get_frequencies()/units.MHz
                        spec = np.abs(channel[1].get_frequency_spectrum()/units.mV)
                    else:
                        freqs = channel[0].get_frequencies()/units.MHz
                        spec = np.zeros(len(freqs))
                    for polarization in range(0,3):
                        if 'direct' in signal_types:
                            name = ''
                        else:
                            name = 'Channel {} ({})'.format(i_channel, polarizaiton_names[polarization])
                        fig.append_trace(
                            go.Scatter(
                                x=freqs,
                                y=spec,
                                opacity=0.5,
                                legendgroup=str(i_channel),
                                name = name,
                                line = {
                                    'dash': linestyles[polarization],
                                    'color': colors[i_channel % len(colors)]
                                },
                                visible=visibility_settings[polarization]
                            ), 1, 1
                        )
        else:
            for polarization in range(0,3):
                fig.append_trace(
                    go.Scatter(
                        x=sim_station.get_frequencies()/units.MHz,
                        y=np.abs(sim_station.get_frequency_spectrum()[polarization])/units.mV,
                        opacity=0.7,
                        line = {
                            'color': colors[polarization % len(colors)]
                        },
                        name = str(polarizaiton_names[polarization]),
                        visible=visibility_settings[polarization]
                    ), 1, 1
                )
    except:
        return {}
    fig['layout'].update(
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
    data = [go.Scatter3d(
        x = [0],
        y = [0],
        z = [0],
        mode = 'markers',
        name = 'Station'
        )]
    if sim_station.has_parameter(stnp.nu_vertex):
        vertex = sim_station.get_parameter(stnp.nu_vertex)
        data.append(go.Scatter3d(
            x = [vertex[0]],
            y = [vertex[1]],
            z = [vertex[2]],
            mode = 'markers',
            name = 'Interaction Vertex'
            ))
        if sim_station.has_parameter(stnp.nu_zenith) and sim_station.has_parameter(stnp.nu_azimuth): 
            neutrino_path = hp.spherical_to_cartesian(sim_station.get_parameter(stnp.nu_zenith), sim_station.get_parameter(stnp.nu_azimuth))
            data.append(go.Scatter3d(
                x = [vertex[0], vertex[0] + 500*neutrino_path[0]],
                y = [vertex[1], vertex[1] + 500*neutrino_path[1]],
                z = [vertex[2], vertex[2] + 500*neutrino_path[2]],
                name = 'Neutrino Direction',
                mode = 'lines'
                ))
    fig = go.Figure(
        data = data,
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
    
@app.callback(Output('sim-station-properties-dropdown', 'options'), 
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value')],
    [State('user_id', 'children'),
     State('station_id', 'children')])
def get_sim_station_property_options(i_event, filename, juser_id, jstation_id):
    if filename is None:
        return []
    user_id = json.loads(juser_id)
    station_id = json.loads(jstation_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_stations()[0]
    options = []
    for parameter in stnp:
        if station.has_parameter(parameter):
            options.append({'label': parameter.name, 'value': parameter.value})
    return options
@app.callback(Output('sim-station-properties-table', 'children'),
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value'),
    Input('sim-station-properties-dropdown', 'value')],
    [State('user_id', 'children'),
     State('station_id', 'children')])
def get_sim_station_property_table(i_event, filename, properties, juser_id, jstation_id):
    if filename is None:
         return []
    user_id = json.loads(juser_id)
    station_id = json.loads(jstation_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_stations()[0]
    reply = []
    for property in properties:
        reply.append(
            html.Div([
                html.Div(str(stnp(property).name), className='custom-table-td'),
                html.Div(str(station.get_parameter(stnp(property))), className='custom-table-td custom-table-td-last')
            ],className='custom-table-row')
        )
    return reply
    