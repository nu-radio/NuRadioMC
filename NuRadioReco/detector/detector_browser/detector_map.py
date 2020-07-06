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
        pos = detector.get_absolute_position(station_id)
        xx.append(pos[0]/units.km)
        yy.append(pos[1]/units.km)
        labels.append(station_id)
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
    for channel_id in channel_ids:
        channel_positions.append(detector.get_relative_position(station_id, channel_id))
    channel_positions = np.array(channel_positions)
    data = []
    data.append(go.Scatter3d(
        x = channel_positions[:,0],
        y = channel_positions[:,1],
        z = channel_positions[:,2],
        ids = channel_ids,
        text = channel_ids,
        mode = 'markers+text',
        textposition = 'middle right',
        marker = dict(
            size=2
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
