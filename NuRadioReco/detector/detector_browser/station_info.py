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
        html.Div('Station Info', className='panel panel-heading'),
        html.Div([
            html.Div('', id='station-info-table')
        ], className='panel panel-body')
    ], className='panel panel-default')
])

@app.callback(
    Output('station-info-table', 'children'),
    [Input('selected-station', 'children')]
)
def update_station_info_table(station_id):
    detector_provider = DetectorProvider()
    detector = detector_provider.get_detector()
    if detector is None:
        return ''
    station_info = detector.get_station(station_id)
    table_rows = [
        html.Div(
            'Station {}'.format(station_id),
            className='custom-table-header'
            )
    ]
    for key, value in station_info.items():
        table_rows.append(html.Div([
            html.Div(key, className='custom-table-title'),
            html.Div(value, className='custom-table-cell')
        ], className='custom-table-row'))
    return table_rows
