import numpy as np
from app import app
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from open_file import DetectorProvider
from NuRadioReco.utilities import units

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
                html.Div(None, id='selected-channel', style={'display': 'none'}),
                dcc.Graph(id='station-view')
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
        )
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
    [Input('selected-station', 'children')]
)
def draw_station_view(station_id):
    if station_id is None:
        return go.Figure([])
    detector_provider = DetectorProvider()
    detector = detector_provider.get_detector()
    channel_positions = []
    channel_ids = detector.get_channel_ids(station_id)
    antenna_types = []
    for channel_id in channel_ids:
        channel_positions.append(detector.get_relative_position(station_id, channel_id))
        antenna_types.append(detector.get_antenna_type(station_id, channel_id))
    channel_positions = np.array(channel_positions)
    antenna_types = np.array(antenna_types)
    data = []
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
    fig = go.Figure(data)
    return fig

@app.callback(
    Output('selected-channel', 'children'),
    [Input('station-view', 'clickData')]
)
def select_channel(click):
    if click is None:
        return None
    return click['points'][0]['id']
