from __future__ import absolute_import, division, print_function  # , unicode_literals
from dash.dependencies import Input, Output, State
import dash
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
            ],style={'flex': '1'}),
        ], className='panel-body', style={'display': 'flex', 'min-height': '500px'})
        ], className='panel panel-default mb-2')
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
