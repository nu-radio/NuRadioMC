import dash
import json
import plotly
from app import app
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import eventParameters as evp
import radiotools.helper as hp
import dataprovider
import numpy as np
provider = dataprovider.DataProvider()

layout = [
    dcc.Graph(id='event-overview', style={'flex': '1'})
]


def get_cherenkov_cone(medium, vertex, nu_dir):
    d_phi = 20*units.deg
    phi_angles = np.arange(0, 360*units.deg, d_phi)
    cherenkov_points = []
    r_cherenkov = 300
    cherenkov_angle = np.arccos(1./medium.get_index_of_refraction(vertex))
    rotation_matrix = hp.get_rotation(np.array([0,0,1]), -1*nu_dir)
    i=[]
    j=[]
    k=[]
    n = 0
    for angle in phi_angles:
        p1 = np.array([1*np.cos(angle), 1*np.sin(angle), 0])
        p1 = rotation_matrix.dot(p1) + vertex
        p2 = np.array([r_cherenkov*np.sin(cherenkov_angle)*np.cos(angle+d_phi), r_cherenkov*np.sin(cherenkov_angle)*np.sin(angle+d_phi), r_cherenkov*np.cos(cherenkov_angle)])
        p2 = rotation_matrix.dot(p2) + vertex
        p3 = np.array([r_cherenkov*np.sin(cherenkov_angle)*np.cos(angle), r_cherenkov*np.sin(cherenkov_angle)*np.sin(angle), r_cherenkov*np.cos(cherenkov_angle)])
        p3 = rotation_matrix.dot(p3) + vertex
        cherenkov_points.append(p1)
        cherenkov_points.append(p2)
        cherenkov_points.append(p3)
        i.append(n)
        j.append(n+1)
        k.append(n+2)
        n += 3
    return np.array(cherenkov_points), i, j, k

@app.callback(Output('event-overview', 'figure'),
              [Input('event-counter-slider', 'value'),
               Input('filename', 'value'),
               Input('station-id-dropdown', 'value')],
              [State('user_id', 'children')])
def plot_event_overview(evt_counter, filename, station_id, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    plots = []
    if station.is_neutrino():
        if station.has_sim_station():
            sim_station = station.get_sim_station()
            sim_vertex = sim_station.get_parameter(stnp.nu_vertex)
            sim_zenith = sim_station.get_parameter(stnp.nu_zenith)
            sim_azimuth = sim_station.get_parameter(stnp.nu_azimuth)
            sim_direction = hp.spherical_to_cartesian(sim_zenith, sim_azimuth)*300.+sim_vertex
            plots.append(plotly.graph_objs.Scatter3d(
                x=[sim_vertex[0]],
                y=[sim_vertex[1]],
                z=[sim_vertex[2]],
                text=['Sim Vertex'],
                mode='markers',
                marker={'size':3, 'color':'red'},
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
            if evt.has_parameter(evp.sim_config) and ariio.get_detector() is not None:
                det = ariio.get_detector()
                det.update(station.get_station_time())
                sim_config = evt.get_parameter(evp.sim_config)
                import NuRadioMC.SignalProp.analyticraytracing
                import NuRadioMC.utilities.medium
                attenuation_model = sim_config['propagation']['attenuation_model']
                ice_model = NuRadioMC.utilities.medium.get_ice_model(sim_config['propagation']['ice_model'])
                n_reflections = sim_config['propagation']['n_reflections']
                for channel_id in det.get_channel_ids(station_id):
                    channel_pos = det.get_relative_position(station_id, channel_id) + det.get_absolute_position(station_id)
                    raytracer = NuRadioMC.SignalProp.analyticraytracing.ray_tracing(sim_vertex, channel_pos, ice_model, attenuation_model, n_reflections=n_reflections)
                    raytracer.find_solutions()
                    for iS,solution in enumerate(raytracer.get_results()):
                        path = raytracer.get_path(iS, 100)
                        plots.append(plotly.graph_objs.Scatter3d(
                            x=path[:,0],
                            y=path[:,1],
                            z=path[:,2],
                            mode='lines',
                            line={'color': 'orange'},
                            showlegend=False,
                            name='Ch. {}, {}'.format(channel_id, NuRadioMC.SignalProp.analyticraytracing.solution_types[solution['type']])
                        ))
                cherenkov_cone, i_ch, j_ch, k_ch = get_cherenkov_cone(ice_model, sim_vertex, hp.spherical_to_cartesian(sim_zenith, sim_azimuth))
                plots.append(plotly.graph_objs.Mesh3d(
                    x=cherenkov_cone[:,0],
                    y=cherenkov_cone[:,1],
                    z=cherenkov_cone[:,2],
                    i=i_ch,
                    j=j_ch,
                    k=k_ch,
                    color='red',
                    opacity=.5,
                    name='Cherenkov ring'))
    if ariio.get_detector() is not None:
        det = ariio.get_detector()
        det.update(station.get_station_time())
        channel_positions = []
        channel_comments = []
        for channel_id in det.get_channel_ids(station_id):
            channel_comments.append(det.get_channel(station_id, channel_id)['ant_comment'])
            channel_positions.append(det.get_absolute_position(station_id) + det.get_relative_position(station_id, channel_id))
        channel_positions = np.array(channel_positions)
        plots.append(plotly.graph_objs.Scatter3d(
        x=channel_positions[:,0],
        y=channel_positions[:,1],
        z=channel_positions[:,2],
        text=channel_comments,
        mode='markers',
        marker={'size': 3},
        showlegend=False,
        name='Station {}'.format(station_id)
        ))
    fig = plotly.graph_objs.Figure(data=plots)
    return fig
