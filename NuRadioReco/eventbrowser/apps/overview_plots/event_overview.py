import json
import plotly
from NuRadioReco.eventbrowser.app import app
from dash.dependencies import Input, Output, State
from dash import html
from dash import dcc
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import eventParameters as evp
from NuRadioReco.framework.parameters import showerParameters as shp
import radiotools.helper as hp
import NuRadioReco.eventbrowser.dataprovider
import numpy as np
import logging

logger = logging.getLogger('overview')

provider = NuRadioReco.eventbrowser.dataprovider.DataProvider()

layout = [
    html.Div([
        html.Div([
            html.Div('Draw Stations'),
            dcc.RadioItems(
                id='overview-station-mode',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'Only selected', 'value': 'selected'}
                ],
                value='selected'
            ),
            dcc.Checklist(
                id='overview-channel-station',
                options=[
                    {'label': 'Station', 'value': 'station'},
                    {'label': 'Channel', 'value': 'channel'}
                ],
                value=['station', 'channel']
            )
        ], style={'flex': 1}),
        html.Div([
            dcc.Graph(id='event-overview', style={'flex': '1'})
        ], style={'flex': 2})
    ], style={'display': 'flex'})
]


def get_cherenkov_cone(medium, vertex, nu_dir):
    d_phi = 20 * units.deg
    phi_angles = np.arange(0, 360 * units.deg, d_phi)
    cherenkov_points = []
    r_cherenkov = 300
    cherenkov_angle = np.arccos(1. / medium.get_index_of_refraction(vertex))
    rotation_matrix = hp.get_rotation(np.array([0, 0, 1]), -1 * nu_dir)
    i = []
    j = []
    k = []
    n = 0
    for angle in phi_angles:
        p1 = np.array([1 * np.cos(angle), 1 * np.sin(angle), 0])
        p1 = rotation_matrix.dot(p1) + vertex
        p2 = np.array([r_cherenkov * np.sin(cherenkov_angle) * np.cos(angle + d_phi),
                       r_cherenkov * np.sin(cherenkov_angle) * np.sin(angle + d_phi),
                       r_cherenkov * np.cos(cherenkov_angle)])
        p2 = rotation_matrix.dot(p2) + vertex
        p3 = np.array([r_cherenkov * np.sin(cherenkov_angle) * np.cos(angle),
                       r_cherenkov * np.sin(cherenkov_angle) * np.sin(angle), r_cherenkov * np.cos(cherenkov_angle)])
        p3 = rotation_matrix.dot(p3) + vertex
        cherenkov_points.append(p1)
        cherenkov_points.append(p2)
        cherenkov_points.append(p3)
        i.append(n)
        j.append(n + 1)
        k.append(n + 2)
        n += 3
    return np.array(cherenkov_points), i, j, k


@app.callback(Output('event-overview', 'figure'),
              [Input('event-counter-slider', 'value'),
               Input('filename', 'value'),
               Input('station-id-dropdown', 'value'),
               Input('overview-station-mode', 'value'),
               Input('overview-channel-station', 'value')],
              [State('user_id', 'children')])
def plot_event_overview(evt_counter, filename, station_id, station_mode, channel_station, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    nurio = provider.get_file_handler(user_id, filename)
    evt = nurio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    plots = []
    # First check the particle type
    try:
        event_is_neutrino = station.is_neutrino() # This throws an error if the particle type has not been set during reconstruction
        event_is_cosmic_ray = station.is_cosmic_ray()
    except ValueError:
        try:
            sim_station = station.get_sim_station()
            event_is_neutrino = sim_station.is_neutrino()
            event_is_cosmic_ray = sim_station.is_cosmic_ray()
        except (AttributeError, ValueError):
            logger.warning('Particle type has not been set in both the station and the sim_station.')
            event_is_neutrino = None
            event_is_cosmic_ray = None
    
    if event_is_neutrino:
        if station.has_sim_station():
            sim_station = station.get_sim_station()
            if sim_station.has_parameter(stnp.nu_vertex): # for backwards compatibility, we attempt to get this from the sim_station
                sim_vertex = sim_station.get_parameter(stnp.nu_vertex)
                sim_zenith = sim_station.get_parameter(stnp.nu_zenith)
                sim_azimuth = sim_station.get_parameter(stnp.nu_azimuth)
            else:
                # we only look at the first sim_shower -> problem if an event contains multiple, different sim_showers?
                sim_showers = [sim_shower for sim_shower in evt.get_sim_showers()]
                sim_vertex = sim_showers[0].get_parameter(shp.vertex)
                sim_zenith = sim_showers[0].get_parameter(shp.zenith)
                sim_azimuth = sim_showers[0].get_parameter(shp.azimuth)
            sim_direction = hp.spherical_to_cartesian(sim_zenith, sim_azimuth) * 300. + sim_vertex
            plots.append(plotly.graph_objs.Scatter3d(
                x=[sim_vertex[0]],
                y=[sim_vertex[1]],
                z=[sim_vertex[2]],
                text=['Sim Vertex'],
                mode='markers',
                marker={'size': 3, 'color': 'red'},
                showlegend=False,
                name='Vertex'
            ))
            plots.append(plotly.graph_objs.Scatter3d(
                x=[sim_vertex[0], sim_direction[0]],
                y=[sim_vertex[1], sim_direction[1]],
                z=[sim_vertex[2], sim_direction[2]],
                mode='lines',
                line={'color': 'red'},
                showlegend=False,
                name='Neutrino direction'
            ))
            if evt.has_parameter(evp.sim_config) and nurio.get_detector() is not None:
                det = nurio.get_detector()
                det.update(station.get_station_time())
                sim_config = evt.get_parameter(evp.sim_config)
                import NuRadioMC.SignalProp.analyticraytracing
                import NuRadioMC.utilities.medium
                attenuation_model = sim_config['propagation']['attenuation_model']
                ice_model = NuRadioMC.utilities.medium.get_ice_model(sim_config['propagation']['ice_model'])
                n_reflections = sim_config['propagation']['n_reflections']
                for channel_id in det.get_channel_ids(station_id):
                    channel_pos = det.get_relative_position(station_id, channel_id) + det.get_absolute_position(
                        station_id)
                    raytracer = NuRadioMC.SignalProp.analyticraytracing.ray_tracing(sim_vertex, channel_pos, ice_model,
                                                                                    attenuation_model,
                                                                                    n_reflections=n_reflections)
                    raytracer.find_solutions()
                    for iS, solution in enumerate(raytracer.get_results()):
                        path = raytracer.get_path(iS, 100)
                        plots.append(plotly.graph_objs.Scatter3d(
                            x=path[:, 0],
                            y=path[:, 1],
                            z=path[:, 2],
                            mode='lines',
                            line={'color': 'orange'},
                            showlegend=False,
                            name='Ch. {}, {}'.format(channel_id, NuRadioMC.SignalProp.analyticraytracing.solution_types[
                                solution['type']])
                        ))
                cherenkov_cone, i_ch, j_ch, k_ch = get_cherenkov_cone(ice_model, sim_vertex,
                                                                      hp.spherical_to_cartesian(sim_zenith,
                                                                                                sim_azimuth))
                plots.append(plotly.graph_objs.Mesh3d(
                    x=cherenkov_cone[:, 0],
                    y=cherenkov_cone[:, 1],
                    z=cherenkov_cone[:, 2],
                    i=i_ch,
                    j=j_ch,
                    k=k_ch,
                    color='red',
                    opacity=.5,
                    name='Cherenkov ring'))
    if event_is_cosmic_ray:
        for sim_shower in evt.get_sim_showers():
            sim_core = sim_shower.get_parameter(shp.core)
            sim_zenith = sim_shower.get_parameter(shp.zenith)
            sim_azimuth = sim_shower.get_parameter(shp.azimuth)
            sim_direction = hp.spherical_to_cartesian(sim_zenith, sim_azimuth)
            plots.append(plotly.graph_objs.Scatter3d(
                x=[sim_core[0]],
                y=[sim_core[1]],
                z=[sim_core[2]],
                mode='markers',
                marker={'color': 'red'},
                showlegend=False,
                name='Core position'
            ))
            dir_points = np.array([sim_core, sim_core + 100 * sim_direction])
            plots.append(plotly.graph_objs.Scatter3d(
                x=dir_points[:, 0],
                y=dir_points[:, 1],
                z=dir_points[:, 2],
                mode='lines',
                line={'color': 'red'},
                showlegend=False,
                name='Shower direction'
            ))

    if nurio.get_detector() is not None:
        det = nurio.get_detector()
        det.update(station.get_station_time())
        channel_positions = []
        channel_comments = []
        station_positions = []
        station_comments = []
        if station_mode == 'selected':
            draw_stations = [station_id]
        else:
            draw_stations = det.get_station_ids()
        for stat_id in draw_stations:
            station_positions.append(det.get_absolute_position(stat_id))
            station_comments.append('Station {}'.format(stat_id))
            for channel_id in det.get_channel_ids(stat_id):
                channel_comments.append(
                    'Ch.{}[{}]'.format(channel_id, det.get_channel(stat_id, channel_id)['ant_comment']))
                channel_positions.append(
                    det.get_absolute_position(stat_id) + det.get_relative_position(stat_id, channel_id))
        channel_positions = np.array(channel_positions)
        station_positions = np.array(station_positions)
        if 'station' in channel_station:
            plots.append(plotly.graph_objs.Scatter3d(
                x=station_positions[:, 0],
                y=station_positions[:, 1],
                z=station_positions[:, 2],
                text=station_comments,
                mode='markers',
                marker={'size': 4},
                showlegend=False,
                name='Stations'
            ))
        if 'channel' in channel_station:
            plots.append(plotly.graph_objs.Scatter3d(
                x=channel_positions[:, 0],
                y=channel_positions[:, 1],
                z=channel_positions[:, 2],
                text=channel_comments,
                mode='markers',
                marker={'size': 3},
                showlegend=False,
                name='Channels'
            ))
    fig = plotly.graph_objs.Figure(data=plots)
    return fig
