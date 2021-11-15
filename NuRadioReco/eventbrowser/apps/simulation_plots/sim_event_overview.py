import json
import plotly
import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
import radiotools.helper as hp
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from NuRadioReco.eventbrowser.app import app
import NuRadioReco.eventbrowser.dataprovider
import logging

logger = logging.getLogger('traces')
provider = NuRadioReco.eventbrowser.dataprovider.DataProvider()

layout = [
    dcc.Graph(id='sim-event-3d', style={'flex': '1'}),
    html.Div([
        dcc.Dropdown(id='sim-station-properties-dropdown', options=[], multi=True, value=[]),
        html.Div(id='sim-station-properties-table', className='table table-striped')
    ], style={'flex': '1', 'min-height': '500px'})
]


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
    nurio = provider.get_file_handler(user_id, filename)
    evt = nurio.get_event_i(i_event)
    station = evt.get_station(station_id)
    det = nurio.get_detector()
    det.update(station.get_station_time())
    sim_station = station.get_sim_station()
    sim_showers = [sim_shower for sim_shower in evt.get_sim_showers()]
    if sim_station is None:
        logger.info("No simulated station for selected event and station")
        return {}
    data = [plotly.graph_objs.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        name='Station'
    )]
    vertex, neutrino_path = None, None
    if sim_station.has_parameter(stnp.nu_vertex): # look for the neutrino vertex, direction in the sim_station
        vertex = sim_station.get_parameter(stnp.nu_vertex)
        event_type = 'Neutrino'
        if sim_station.has_parameter(stnp.nu_zenith) and sim_station.has_parameter(stnp.nu_azimuth):
            neutrino_path = hp.spherical_to_cartesian(sim_station.get_parameter(stnp.nu_zenith),
                                                      sim_station.get_parameter(stnp.nu_azimuth))
    elif len(sim_showers) > 0: # look in the event sim_showers
        if sim_station.is_neutrino():
            event_type = 'Neutrino'
            vertices = np.unique([ss.get_parameter(shp.vertex) for ss in sim_showers], axis=0)
        else:
            event_type = 'Cosmic Ray'
            vertices = np.unique([ss.get_parameter(shp.core) for ss in sim_showers], axis=0)
        zeniths = np.unique([ss.get_parameter(shp.zenith) for ss in sim_showers], axis=0)
        azimuths = np.unique([ss.get_parameter(shp.azimuth) for ss in sim_showers], axis=0)
        if any([len(k) > 1 for k in [vertices, zeniths, azimuths]]):
            logger.warning("Event contains more than one shower. Only the first shower will be shown.")
        if len(vertices):
            vertex = vertices[0] - det.get_absolute_position(station_id) # shower vertex coordinates are global coordinates
            if len(zeniths) & len(azimuths):
                neutrino_path = hp.spherical_to_cartesian(zeniths[0], azimuths[0])
    else:
        logger.info("Simulated neutrino vertex not found.")
        plot_range = 1 * units.km
    
    if vertex is not None:
        data.append(plotly.graph_objs.Scatter3d(
                x=[vertex[0]],
                y=[vertex[1]],
                z=[vertex[2]],
                mode='markers',
                name='Interaction Vertex'
            ))
        plot_range = 1.5 * np.max(np.abs(vertex))
        if neutrino_path is not None:
            data.append(plotly.graph_objs.Scatter3d(
                x=[vertex[0], vertex[0] + .25 * plot_range * neutrino_path[0]],
                y=[vertex[1], vertex[1] + .25 * plot_range * neutrino_path[1]],
                z=[vertex[2], vertex[2] + .25 * plot_range * neutrino_path[2]],
                name='{} Direction'.format(event_type),
                mode='lines'
            ))
        
    fig = plotly.graph_objs.Figure(
        data=data,
        layout=plotly.graph_objs.Layout(
            width=1000,
            height=1000,
            legend={
                'orientation': 'h',
                'y': 1.1
            },
            scene={
                'aspectmode': 'manual',
                'aspectratio': {
                    'x': 2,
                    'y': 2,
                    'z': 1
                },
                'xaxis': {
                    'range': [-plot_range, plot_range]
                },
                'yaxis': {
                    'range': [-plot_range, plot_range]
                },
                'zaxis': {
                    'range': [-plot_range, np.max([0, vertex[2] + .3 * plot_range * neutrino_path[2]])]
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
        logger.info('No file or station selected')
        return []
    user_id = json.loads(juser_id)
    nurio = provider.get_file_handler(user_id, filename)
    evt = nurio.get_event_i(i_event)
    station = evt.get_station(station_id).get_sim_station()
    if station is None:
        logger.info('No simulated station found')
        return []
    options = []
    all_params = []
    for parameter in stnp:
        all_params.append(parameter.name)
        if station.has_parameter(parameter):
            options.append({'label': parameter.name, 'value': parameter.value})
    found_params = [option['label'] for option in options]
    not_found_params = [param for param in all_params if param not in found_params]
    logger.info('Simulated station has the following parameters:\n  {}\nNot defined are:\n  {}'.format(found_params, not_found_params))
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
    nurio = provider.get_file_handler(user_id, filename)
    evt = nurio.get_event_i(i_event)
    station = evt.get_station(station_id).get_sim_station()
    reply = []
    for prop in properties:
        reply.append(
            html.Div([
                html.Div(str(stnp(prop).name), className='custom-table-td'),
                html.Div(str(station.get_parameter(stnp(prop))), className='custom-table-td custom-table-td-last')
            ], className='custom-table-row')
        )
    return reply
