import dash
import json
import plotly
from NuRadioReco.utilities import units
from NuRadioReco.eventbrowser.default_layout import default_layout
import numpy as np
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app
import dataprovider
provider = dataprovider.DataProvider()

layout = [
    dcc.Graph(id='time-trace')
]

@app.callback(
    dash.dependencies.Output('time-trace', 'figure'),
    [dash.dependencies.Input('trigger-trace', 'children'),
     dash.dependencies.Input('event-counter-slider', 'value'),
     dash.dependencies.Input('filename', 'value'),
     dash.dependencies.Input('station-id-dropdown', 'value')],
     [State('user_id', 'children')])
def update_time_trace(trigger, evt_counter, filename, station_id, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    traces = []
    fig = plotly.subplots.make_subplots(rows=1, cols=1)
    trace_start_times = []
    for channel in station.iter_channels():
        trace_start_times.append(channel.get_trace_start_time())
    if np.min(trace_start_times) > 1000.*units.ns:
        trace_start_time_offset = np.floor(np.min(trace_start_times)/1000.)*1000.
    else:
        trace_start_time_offset = 0
    for i, channel in enumerate(station.iter_channels()):
        if channel.get_trace() is None:
            continue
        fig.append_trace(plotly.graph_objs.Scatter(
                x=channel.get_times() - trace_start_time_offset / units.ns,
                y=channel.get_trace() / units.mV,
                # text=df_by_continent['country'],
                # mode='markers',
                opacity=0.7,
                marker={
                    'color': colors[i % len(colors)],
                    'line': {'color': colors[i % len(colors)]}
                },
                name='Channel {}'.format(i)
            ), 1, 1)
    fig['layout'].update(default_layout)
    if trace_start_time_offset > 0:
        fig['layout']['xaxis1'].update(title='time [ns] - {:.0f}ns'.format(trace_start_time_offset))
    else:
        fig['layout']['xaxis1'].update(title='time [ns]')
    fig['layout']['yaxis1'].update(title='voltage [mV]')
    return fig
