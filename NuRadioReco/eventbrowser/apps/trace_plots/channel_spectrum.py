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
    dcc.Graph(id='channel-spectrum')
]

@app.callback(
    dash.dependencies.Output('channel-spectrum', 'figure'),
    [dash.dependencies.Input('trigger-trace', 'children'),
     dash.dependencies.Input('event-counter-slider', 'value'),
     dash.dependencies.Input('filename', 'value'),
     dash.dependencies.Input('station-id-dropdown', 'value')],
     [State('user_id', 'children')])
def update_channel_spectrum(trigger, evt_counter, filename, station_id, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    traces = []
    fig = plotly.subplots.make_subplots(rows=1, cols=1)
    maxL1 = 0
    maxY = 0
    ff=np.array([0])
    for i, channel in enumerate(station.iter_channels()):
        if channel.get_trace() is None:
            continue
        tt = channel.get_times()
        dt = tt[1] - tt[0]
        if channel.get_trace() is None:
            continue
        trace = channel.get_trace()
        ff = np.fft.rfftfreq(len(tt), dt)
        spec = np.abs(np.fft.rfft(trace, norm='ortho'))
        fig.append_trace(plotly.graph_objs.Scatter(
                x=ff / units.MHz,
                y=spec / units.mV,
                opacity=0.7,
                marker={
                    'color': colors[i % len(colors)],
                    'line': {'color': colors[i % len(colors)]}
                },
                name='Channel {}'.format(i)
            ), 1, 1)
    fig['layout'].update(default_layout)
    fig['layout']['xaxis1'].update(title='frequency [MHz]')
    fig['layout']['yaxis1'].update(title='amplitude [mV]')
    return fig
