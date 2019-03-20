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
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.eventbrowser.default_layout import default_layout
import numpy as np
import logging
logger = logging.getLogger('traces')

polarizaiton_names = ['r', 'theta', 'phi']
provider = dataprovider.DataProvider()

efield_plot_colors = [
    ['rgb(91, 179, 240)', 'rgb(31, 119, 180)'],
    ['rgb(255, 187, 94)', 'rgb(255, 127, 14)'],
    ['rgb(104, 220, 104)', 'rgb(44, 160, 44)'],
    ['rgb(255, 99, 100)', 'rgb(214, 39, 40)'],
    ['rgb(208, 163, 249)', 'rgb(148, 103, 189)'],
    ['rgb(200, 146, 135)', 'rgb(140, 86, 75)']
]
efield_plot_linestyles = {
    'direct': 'solid',
    'reflected': 'dash',
    'refracted': 'dot'
}

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
                                    {'label': 'reflected', 'value': 'reflected'},
                                    {'label': 'refracted', 'value': 'refracted'}
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
                            html.Div('Signal Type', className=''),
                            dcc.Checklist(
                                id='sim-spectrum-signal-types',
                                options = [
                                    {'label': 'direct', 'value': 'direct'},
                                    {'label': 'reflected', 'value': 'reflected'},
                                    {'label': 'refracted', 'value': 'refracted'}
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
            ], className='panel panel-default', style = {'flex': '1'}),
            html.Div([
                html.Div('Reconstruction Quality', className='panel-heading'),
                html.Div([
                    html.Div([
                        html.Div('Property:', style={'flex': '1', 'padding': '0 10px'}),
                        html.Div([
                            dcc.Dropdown(id='reconstruction-quality-properties-dropdown', options=[], multi=False, value=None)
                        ], style={'flex': '5'})
                    ], style={'display': 'flex'}),
                    html.Div([
                        dcc.Graph(id='reconstruction-quality-histogram')
                    ])
                ], className='panel-body')
            ], className='panel panel-default', style={'flex': '1'})
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
    Input('sim-traces-signal-types', 'values'),
    Input('station-id-dropdown', 'value')],
    [State('user_id', 'children')]
)
def update_sim_trace_plot(i_event, filename, signal_types, station_id, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_station(station_id)
    sim_station = station.get_sim_station()
    visibility_settings = ['legendonly', True, True]
    if sim_station is None:
        return {}
    fig = tools.make_subplots(rows=1, cols=1)
    try:
        for i_electric_field, electric_field in enumerate(sim_station.get_electric_fields()):
            if electric_field.get_parameter(efp.ray_path_type) in signal_types:
                for polarization in range(1,3):
                    fig.append_trace(
                        go.Scatter(
                            x=electric_field.get_times()/units.ns,
                            y=electric_field.get_trace()[polarization]/units.mV*units.m,
                            opacity=1.,
                            line = {
                                'color': efield_plot_colors[i_electric_field % len(efield_plot_colors)][polarization-1],
                                'dash': efield_plot_linestyles[electric_field.get_parameter(efp.ray_path_type)]
                            },
                            name = 'Ch. {} {} ({})'.format(electric_field.get_channel_ids(), polarizaiton_names[polarization], electric_field.get_parameter(efp.ray_path_type)),
                            legendgroup=str(i_electric_field),
                            visible=visibility_settings[polarization]
                        ), 1, 1
                    )
    except:
        return {}
    fig['layout'].update(default_layout)
    fig['layout'].update(
        xaxis={'title': 't [ns]'},
        yaxis={'title': 'electric field [mV/m]'}
    )
    return fig

@app.callback(
    Output('sim-spectrum', 'figure'),
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value'),
    Input('sim-spectrum-signal-types', 'values'),
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
    fig = tools.make_subplots(rows=1, cols=1)
    try:
        for i_electric_field, electric_field in enumerate(sim_station.get_electric_fields()):
            if electric_field.get_parameter(efp.ray_path_type) in signal_types:
                for polarization in range(1,3):
                    fig.append_trace(
                        go.Scatter(
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
    except:
        return {}
    fig['layout'].update(default_layout)
    fig['layout'].update(
        xaxis={'title': 'f [MHz]'},
        yaxis={'title': 'electric field [mV/m]'}
    )
    return fig
    
@app.callback(
    Output('sim-event-3d', 'figure'),
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value'),
    Input('station-id-dropdown', 'value')],
    [State('user_id', 'children')]
     )
def update_sim_event_3d(i_event, filename, station_id, juser_id):
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
        plot_range = 1.5*np.max(np.abs(vertex))
        if sim_station.has_parameter(stnp.nu_zenith) and sim_station.has_parameter(stnp.nu_azimuth): 
            neutrino_path = hp.spherical_to_cartesian(sim_station.get_parameter(stnp.nu_zenith), sim_station.get_parameter(stnp.nu_azimuth))
            data.append(go.Scatter3d(
                x = [vertex[0], vertex[0] + .25*plot_range*neutrino_path[0]],
                y = [vertex[1], vertex[1] + .25*plot_range*neutrino_path[1]],
                z = [vertex[2], vertex[2] + .25*plot_range*neutrino_path[2]],
                name = 'Neutrino Direction',
                mode = 'lines'
                ))
    else:
        plot_range = 1*units.km
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
                    'range': [-plot_range,plot_range]
                },
                'yaxis' : {
                    'range': [-plot_range, plot_range]
                },
                'zaxis' : {
                    'range': [-plot_range,0]
                }
                }
            )
        )
    return fig
    
@app.callback(Output('sim-station-properties-dropdown', 'options'), 
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value'),
    Input('station-id-dropdown', 'value')],
    [State('user_id', 'children')])
def get_sim_station_property_options(i_event, filename, station_id, juser_id):
    if filename is None or station_id is None:
        return []
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_station(station_id).get_sim_station()
    if station is None:
        return []
    options = []
    for parameter in stnp:
        if station.has_parameter(parameter):
            options.append({'label': parameter.name, 'value': parameter.value})
    return options
@app.callback(Output('sim-station-properties-table', 'children'),
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value'),
    Input('sim-station-properties-dropdown', 'value'),
    Input('station-id-dropdown', 'value')],
    [State('user_id', 'children')])
def get_sim_station_property_table(i_event, filename, properties, station_id, juser_id):
    if filename is None or station_id is None:
         return []
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_station(station_id).get_sim_station()
    reply = []
    for property in properties:
        reply.append(
            html.Div([
                html.Div(str(stnp(property).name), className='custom-table-td'),
                html.Div(str(station.get_parameter(stnp(property))), className='custom-table-td custom-table-td-last')
            ],className='custom-table-row')
        )
    return reply

@app.callback(Output('reconstruction-quality-properties-dropdown', 'options'),
    [Input('filename', 'value'),
    Input('station-id-dropdown', 'value')],
    [State('user_id', 'children')])
def get_reconstruction_quality_property_options(filename, station_id, juser_id):
    if filename is None or station_id is None:
        return []
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(0)      # we assume that the same properties are set for every event and every station
    station = evt.get_station(station_id)
    sim_station = station.get_sim_station()
    if sim_station is None:
        return []
    options = []
    for parameter in stnp:
        if station.has_parameter(parameter) and sim_station.has_parameter(parameter):
            options.append({'label': parameter.name, 'value': parameter.value})
    return options
    
    
@app.callback(Output('reconstruction-quality-histogram', 'figure'),
              [Input('filename', 'value'),
              Input('reconstruction-quality-properties-dropdown', 'value'),
              Input('event-ids', 'children'),
              Input('station-id-dropdown', 'value')],
              [State('user_id', 'children')])
def plot_reconstruction_quality_histogram(filename, selected_property, jcurrent_selection, station_id, juser_id):
    if filename is None or selected_property is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    current_selection = json.loads(jcurrent_selection)
    e_diffs = []
    in_selection = []
    ariio = provider.get_arianna_io(user_id, filename)
    for i_event,  event in enumerate(ariio.get_events()):
        station = event.get_station(station_id)
        sim_station = station.get_sim_station()
        e_sim = sim_station.get_parameter(stnp(selected_property))
        e_rec = station.get_parameter(stnp(selected_property))
        e_diffs.append(2.*(e_rec-e_sim)/(e_rec+e_sim))
        if len(current_selection) == 0:
            in_selection.append(True)
        else:
            in_selection.append(i_event in current_selection)
    e_diffs = np.array(e_diffs)
    in_selection = np.array(in_selection)
    x_coords = []
    y_coords = []
    y_coords_selected = []
    d_bin = .05
    for i in np.arange(-2,2,d_bin):
        x_coords.append(i+.5*d_bin)
        y_coords.append(len(e_diffs[(e_diffs>i)&(e_diffs<i+d_bin)]))
        y_coords_selected.append(len(e_diffs[in_selection&(e_diffs>i)&(e_diffs<i+d_bin)]))
    quantiles = np.round(np.percentile(e_diffs, [16,84, 50, 2.5, 97.5]),2)
    annotations = [{
        'x': .9,
        'y': .95,
        'yanchor': 'middle',
        'showarrow': False,
        'xref': 'paper',
        'yref': 'paper',
        'text': 'median: {}'.format(quantiles[2]),
        'font': {
            'color': plotly.colors.DEFAULT_PLOTLY_COLORS[0]
        }
    },{
        'x': .9,
        'y': .9,
        'yanchor': 'middle',
        'showarrow': False,
        'xref': 'paper',
        'yref': 'paper',
        'text': '68% quantile: {} to {}'.format(quantiles[0], quantiles[1]),
        'font': {
            'color': plotly.colors.DEFAULT_PLOTLY_COLORS[0]
        }
    },{
        'x': .9,
        'y': .85,
        'yanchor': 'middle',
        'showarrow': False,
        'xref': 'paper',
        'yref': 'paper',
        'text': '95% quantile: {} to {}'.format(quantiles[3], quantiles[4]),
        'font': {
            'color': plotly.colors.DEFAULT_PLOTLY_COLORS[0]
        }
    }]
    if not np.all(in_selection):
        quantiles_selected = np.round(np.percentile(e_diffs[in_selection], [16,84, 50, 2.5, 97.5]),2)
        annotations = np.append(annotations, [{
            'x': .9,
            'y': .75,
            'yanchor': 'middle',
            'showarrow': False,
            'xref': 'paper',
            'yref': 'paper',
            'text': 'median: {}'.format(quantiles_selected[2]),
            'font': {
                'color': plotly.colors.DEFAULT_PLOTLY_COLORS[1]
            }
        },{
            'x': .9,
            'y': .7,
            'yanchor': 'middle',
            'showarrow': False,
            'xref': 'paper',
            'yref': 'paper',
            'text': '68% quantile: {} to {}'.format(quantiles_selected[0], quantiles_selected[1]),
            'font': {
                'color': plotly.colors.DEFAULT_PLOTLY_COLORS[1]
            }
        },{
            'x': .9,
            'y': .65,
            'yanchor': 'middle',
            'showarrow': False,
            'xref': 'paper',
            'yref': 'paper',
            'text': '95% quantile: {} to {}'.format(quantiles_selected[3], quantiles_selected[4]),
            'font': {
                'color': plotly.colors.DEFAULT_PLOTLY_COLORS[1]
            }
        }])
    layout = {
        'xaxis': {
            'title': 'Relative error of reconstructed property'
        },
        'yaxis': {
            'title': 'Entries'
        },
        'annotations': annotations
    }
    if np.all(in_selection):
        return {
            'data': [{
                'x': x_coords,
                'y': y_coords,
                'type': 'bar',
                'marker': {
                    'color': plotly.colors.DEFAULT_PLOTLY_COLORS[0]
                }
            }],
            'layout': layout
        }
    else:
        return {
            'data': [{
                'x': x_coords,
                'y': y_coords,
                'type': 'bar',
                'name': 'All events',
                'marker': {
                    'color': plotly.colors.DEFAULT_PLOTLY_COLORS[0]
                }
            },{
                'x': x_coords,
                'y': y_coords_selected,
                'type': 'bar',
                'name': 'Selected events',
                'marker': {
                    'color': plotly.colors.DEFAULT_PLOTLY_COLORS[1]
                }
            }],
            'layout': layout
        }
