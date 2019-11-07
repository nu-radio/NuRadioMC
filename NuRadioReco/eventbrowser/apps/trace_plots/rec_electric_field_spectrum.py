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
dcc.Graph(id='efield-spectrum')
]

@app.callback(
    dash.dependencies.Output('efield-spectrum', 'figure'),
    [dash.dependencies.Input('trigger-trace', 'children'),
     dash.dependencies.Input('event-counter-slider', 'value'),
     dash.dependencies.Input('filename', 'value'),
     dash.dependencies.Input('station-id-dropdown', 'value')],
     [State('user_id', 'children')])
def update_efield_spectrum(trigger, evt_counter, filename, station_id, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    fig = plotly.subplots.make_subplots(rows=1, cols=1)
    for electric_field in station.get_electric_fields():
        if electric_field.get_frequencies() is None:
            spectrum = np.array([[],[],[]])
            frequencies = np.array([[],[],[]])
        else:
            spectrum = electric_field.get_frequency_spectrum()
            frequencies = electric_field.get_frequencies()
        fig.append_trace(plotly.graph_objs.Scatter(
                x=frequencies / units.MHz,
                y=np.abs(spectrum[1]) / units.mV,
                opacity=0.7,
                marker={
                    'color': colors[1],
                    'line': {'color': colors[1]}
                },
                name='eTheta'
            ), 1, 1)
        fig.append_trace(plotly.graph_objs.Scatter(
                x=frequencies / units.MHz,
                y=np.abs(spectrum[2]) / units.mV,
                opacity=0.7,
                marker={
                    'color': colors[2],
                    'line': {'color': colors[2]}
                },
                name='ePhi'
            ), 1, 1)
    fig['layout'].update(default_layout)
    fig['layout']['xaxis1'].update(title='frequency [MHz]')
    fig['layout']['yaxis1'].update(title='amplitude [mV/m]')
    return fig
