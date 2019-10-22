import dash
import json
import plotly
from NuRadioReco.utilities import units
from NuRadioReco.eventbrowser.default_layout import default_layout
import numpy as np

def update_efield_spectrum(trigger, evt_counter, filename, station_id, juser_id, provider):
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
