import dash
import json
import plotly.subplots
import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.eventbrowser.default_layout import default_layout
from dash import dcc
from dash.dependencies import Input, Output, State
from NuRadioReco.eventbrowser.app import app
import NuRadioReco.eventbrowser.dataprovider

provider = NuRadioReco.eventbrowser.dataprovider.DataProvider()

layout = [
    dcc.Graph(id='efield-trace')
]


@app.callback(
    Output('efield-trace', 'figure'),
    [Input('trigger-trace', 'children'),
     Input('event-counter-slider', 'value'),
     Input('filename', 'value'),
     Input('station-id-dropdown', 'value')],
    [State('user_id', 'children')])
def update_time_efieldtrace(trigger, evt_counter, filename, station_id, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    nurio = provider.get_file_handler(user_id, filename)
    evt = nurio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    fig = plotly.subplots.make_subplots(rows=1, cols=1)
    for electric_field in station.get_electric_fields():
        if electric_field.get_trace() is None:
            trace = np.array([[], [], []])
        else:
            trace = electric_field.get_trace()
        fig.append_trace(plotly.graph_objs.Scatter(
            x=electric_field.get_times() / units.ns,
            y=trace[0] / units.mV * units.m,
            # text=df_by_continent['country'],
            # mode='markers',
            opacity=0.7,
            marker={
                'color': colors[0],
                'line': {'color': colors[0]}
            },
            name='eR'
        ), 1, 1)
        fig.append_trace(plotly.graph_objs.Scatter(
            x=electric_field.get_times() / units.ns,
            y=trace[1] / units.mV * units.m,
            # text=df_by_continent['country'],
            # mode='markers',
            opacity=0.7,
            marker={
                'color': colors[1],
                'line': {'color': colors[1]}
            },
            name='eTheta'
        ), 1, 1)
        fig.append_trace(plotly.graph_objs.Scatter(
            x=electric_field.get_times() / units.ns,
            y=trace[2] / units.mV * units.m,
            # text=df_by_continent['country'],
            # mode='markers',
            opacity=0.7,
            marker={
                'color': colors[2],
                'line': {'color': colors[2]}
            },
            name='ePhi'
        ), 1, 1)
    fig['layout']['xaxis1'].update(title='time [ns]')
    fig['layout']['yaxis1'].update(title='efield [mV/m]')
    fig['layout'].update(default_layout)
    return fig
