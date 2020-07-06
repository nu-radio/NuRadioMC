import numpy as np
from app import app
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from open_file import DetectorProvider

layout = html.Div([
    html.Div([
        html.Div('Channel Info', className='panel panel-heading'),
        html.Div([
            html.Div('', id='channel-info-table')
        ], className='panel panel-body')
    ], className='panel panel-default')
])

@app.callback(
    Output('channel-info-table', 'children'),
    [Input('selected-station', 'children'),
    Input('selected-channel', 'children')]
)
def update_channel_info_table(station_id, channel_id):
    detector_provider = DetectorProvider()
    detector = detector_provider.get_detector()
    if detector is None:
        return ''
    if channel_id not in detector.get_channel_ids(station_id):
        print('channel not in station', channel_id)
        return ''
    channel_info = detector.get_channel(station_id, channel_id)
    table_rows = [
        html.Div(
            'Station {}, Channel {}'.format(station_id, channel_id),
            className='custom-table-header'
        )
    ]
    for key, value in channel_info.items():
        table_rows.append(html.Div([
            html.Div(key, className='custom-table-title'),
            html.Div(value, className='custom-table-cell')
        ], className='custom-table-row'))
    return table_rows
