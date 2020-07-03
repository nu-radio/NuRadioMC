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
        html.Div('Station Map', className='panel panel-heading'),
        html.Div([
            html.Div(None, id='selected-station', style={'display': 'none'}),
            dcc.Graph(id='station-position-map')
        ], className='panel panel-body')
    ], className='panel panel-default')
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
