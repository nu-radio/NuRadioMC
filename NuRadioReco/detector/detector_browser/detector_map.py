import numpy as np
from app import app
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from open_file import DetectorProvider
from NuRadioReco.utilities import units
import radiotools.helper as hp

layout = html.Div([
    html.Div([
        html.Div([
            html.Div('Station Map', className='panel panel-heading'),
            html.Div([
                html.Div(None, id='selected-station', style={'display': 'none'}),
                dcc.Graph(id='station-position-map')
            ], className='panel panel-body')
        ], className='panel panel-default', style={'flex': '1'}),
        html.Div([
            html.Div('Station View', className='panel panel-heading'),
            html.Div([
                dcc.Graph(id='station-view'),
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id='station-view-checklist',
                            options=[
                                {'label': 'Antenna Sketch', 'value': 'sketch'}
                            ],
                            value=[],
                            labelStyle={'padding': '2px'}
                        )
                    ], className='option-select')
                ], className='option-set'),
                html.Div(None, id='selected-channel', style={'display': 'none'})
            ], className='panel panel-body')
        ], className='panel panel-default', style={'flex': '1'})
    ], style={'display': 'flex'})
])

@app.callback(
    Output('station-position-map', 'figure'),
    [Input('output-dummy', 'children')]
)
def draw_station_position_map(dummy):
    detector_provider = DetectorProvider()
    detector = detector_provider.get_detector()
    if detector is None:
        return go.Figure([])
    xx = []
    yy = []
    labels = []
    for station_id in detector.get_station_ids():
        try:
            pos = detector.get_absolute_position(station_id)
            xx.append(pos[0]/units.km)
            yy.append(pos[1]/units.km)
            labels.append(station_id)
        except:
            continue
    data = [
        go.Scatter(
            x=xx,
            y=yy,
            ids=labels,
            text=labels,
            mode='markers+text',
            textposition='middle right'

        )
    ]
    fig = go.Figure(data)
    fig.update_layout(
        xaxis = dict(
            title='Easting [km]'
        ),
        yaxis = dict(
            title='Northing [km]',
            scaleanchor = 'x',
            scaleratio = 1
        ),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

@app.callback(
    Output('selected-station', 'children'),
    [Input('station-position-map', 'clickData')]
)
def select_station(click):
    if click is None:
        return None
    return click['points'][0]['id']

@app.callback(
    Output('station-view', 'figure'),
    [Input('selected-station', 'children'),
    Input('station-view-checklist', 'value')]
)
def draw_station_view(station_id, checklist):
    if station_id is None:
        return go.Figure([])
    detector_provider = DetectorProvider()
    detector = detector_provider.get_detector()
    channel_positions = []
    channel_ids = detector.get_channel_ids(station_id)
    antenna_types = []
    antenna_orientations = []
    antenna_rotations = []
    data = []
    for channel_id in channel_ids:
        channel_position = detector.get_relative_position(station_id, channel_id)
        channel_positions.append(channel_position)
        antenna_types.append(detector.get_antenna_type(station_id, channel_id))
        orientation = detector.get_antenna_orientation(station_id, channel_id)
        ant_ori = hp.spherical_to_cartesian(orientation[0], orientation[1])
        antenna_orientations.append(channel_position)
        antenna_orientations.append(channel_position + ant_ori)
        antenna_orientations.append([None, None, None])
        ant_rot = hp.spherical_to_cartesian(orientation[2], orientation[3])
        antenna_rotations.append(channel_position)
        antenna_rotations.append(channel_position + ant_rot)
        antenna_rotations.append([None, None, None])
        if 'createLPDA' in detector.get_antenna_type(station_id, channel_id) and 'sketch' in checklist:
            antenna_tip = channel_position + ant_ori
            antenna_tine_1 = channel_position + np.cross(ant_ori, ant_rot)
            antenna_tine_2 = channel_position - np.cross(ant_ori, ant_rot)
            data.append(go.Mesh3d(
                x = [antenna_tip[0], antenna_tine_1[0], antenna_tine_2[0]],
                y = [antenna_tip[1], antenna_tine_1[1], antenna_tine_2[1]],
                z = [antenna_tip[2], antenna_tine_1[2], antenna_tine_2[2]],
                opacity=.5,
                color='black',
                delaunayaxis='x',
                hoverinfo='skip'
            ))
    channel_positions = np.array(channel_positions)
    antenna_types = np.array(antenna_types)
    antenna_orientations = np.array(antenna_orientations)
    antenna_rotations = np.array(antenna_rotations)
    lpda_mask = (np.char.find(antenna_types, 'createLPDA')>=0)
    vpol_mask = (np.char.find(antenna_types, 'bicone_v8')>=0)|(np.char.find(antenna_types, 'greenland_vpol')>=0)
    hpol_mask = (np.char.find(antenna_types, 'fourslot')>=0)
    if len(channel_positions[:,0][lpda_mask]) > 0:
        data.append(go.Scatter3d(
            x = channel_positions[:,0][lpda_mask],
            y = channel_positions[:,1][lpda_mask],
            z = channel_positions[:,2][lpda_mask],
            ids = channel_ids,
            text = channel_ids,
            mode = 'markers+text',
            name='LPDAs',
            textposition = 'middle right',
            marker_symbol='diamond-open',
            marker = dict(
                size=4
            )
        ))
    if len(channel_positions[:,0][vpol_mask]) > 0:
        data.append(go.Scatter3d(
            x = channel_positions[:,0][vpol_mask],
            y = channel_positions[:,1][vpol_mask],
            z = channel_positions[:,2][vpol_mask],
            ids = channel_ids,
            text = channel_ids,
            mode = 'markers+text',
            name='V-pol',
            textposition = 'middle right',
            marker_symbol='x',
            marker = dict(
                size=4
            )
        ))
    if len(channel_positions[:,0][hpol_mask]) > 0:
        data.append(go.Scatter3d(
            x = channel_positions[:,0][hpol_mask],
            y = channel_positions[:,1][hpol_mask],
            z = channel_positions[:,2][hpol_mask],
            ids = channel_ids,
            text = channel_ids,
            mode = 'markers+text',
            name='H-pol',
            textposition = 'middle right',
            marker_symbol='cross',
            marker = dict(
                size=4
            )
        ))
    if len(channel_positions[:,0][(~lpda_mask)&(~vpol_mask)&(~hpol_mask)]) > 0:
        data.append(go.Scatter3d(
            x = channel_positions[:,0][(~lpda_mask)&(~vpol_mask)&(~hpol_mask)],
            y = channel_positions[:,1][(~lpda_mask)&(~vpol_mask)&(~hpol_mask)],
            z = channel_positions[:,2][(~lpda_mask)&(~vpol_mask)&(~hpol_mask)],
            ids = channel_ids,
            text = channel_ids,
            mode = 'markers+text',
            name='other',
            textposition = 'middle right',
            marker = dict(
                size=3
            )
        ))
    if len(channel_positions[:,0]) > 0:
        data.append(go.Scatter3d(
            x = antenna_orientations[:,0],
            y = antenna_orientations[:,1],
            z = antenna_orientations[:,2],
            mode='lines',
            name='Orientations',
            marker_color='red',
            hoverinfo='skip'
        ))
        data.append(go.Scatter3d(
            x = antenna_rotations[:,0],
            y = antenna_rotations[:,1],
            z = antenna_rotations[:,2],
            mode='lines',
            name='Rotations',
            marker_color='blue',
            hoverinfo='skip'
        ))

    fig = go.Figure(data)
    fig.update_layout(
        scene=dict(
            aspectmode='data'
        ),
        legend_orientation='h',
        legend=dict(x=.0, y=1.),
        height=600,
        margin=dict(l=10, r=10, t=30, b=10)

    )
    return fig

@app.callback(
    Output('selected-channel', 'children'),
    [Input('station-view', 'clickData')]
)
def select_channel(click):
    if click is None:
        return None
    return click['points'][0]['id']
