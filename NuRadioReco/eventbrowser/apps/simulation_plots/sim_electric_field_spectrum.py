import dash
import json
import plotly
from NuRadioReco.utilities import units
from NuRadioReco.eventbrowser.default_layout import default_layout, efield_plot_colors, efield_plot_linestyles, polarizaiton_names
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app
import dataprovider
import numpy as np
provider = dataprovider.DataProvider()

layout = [
    html.Div([
        html.Div([
            html.Div('Signal Type', className=''),
            dcc.Checklist(
                id='sim-spectrum-signal-types',
                options = [
                    {'label': 'direct', 'value': 'direct'},
                    {'label': 'reflected', 'value': 'reflected'},
                    {'label': 'refracted', 'value': 'refracted'}
                ],
                value = ['direct'],
                className='sim-trace-option'
            )
        ], className='', id='sim-spectrum-signal-types-container')
    ], className='sim-trace-options'),
    html.Div([
        dcc.Graph(id='sim-spectrum')
    ]),
]

@app.callback(
    Output('sim-spectrum', 'figure'),
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value'),
    Input('sim-spectrum-signal-types', 'value'),
    Input('station-id-dropdown', 'value')],
    [State('user_id', 'children')]
)
def update_sim_spectrum_plot(i_event, filename, signal_types, station_id, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_station(station_id)
    sim_station = station.get_sim_station()
    if sim_station is None:
        return {}
    fig = plotly.subplots.make_subplots(rows=1, cols=1)
    for i_electric_field, electric_field in enumerate(sim_station.get_electric_fields()):
        if electric_field.get_parameter(efp.ray_path_type) in signal_types:
            for polarization in range(1,3):
                fig.append_trace(
                    plotly.graph_objs.Scatter(
                        x=electric_field.get_frequencies()/units.MHz,
                        y=np.abs(electric_field.get_frequency_spectrum()[polarization])/units.mV*units.m,
                        opacity=1.,
                        line = {
                            'color': efield_plot_colors[i_electric_field % len(efield_plot_colors)][polarization-1],
                            'dash': efield_plot_linestyles[electric_field.get_parameter(efp.ray_path_type)]
                        },
                        name = 'Ch. {} {} ({})'.format(electric_field.get_channel_ids(), polarizaiton_names[polarization], electric_field.get_parameter(efp.ray_path_type)),
                        legendgroup=str(i_electric_field)
                    ), 1, 1
                )
    fig['layout'].update(default_layout)
    fig['layout'].update(
        xaxis={'title': 'f [MHz]'},
        yaxis={'title': 'electric field [mV/m]'}
    )
    return fig
