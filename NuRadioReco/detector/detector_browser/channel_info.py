from NuRadioReco.detector.detector_browser.app import app
from dash import html
from dash.dependencies import Input, Output
import NuRadioReco.detector.detector_browser.detector_provider

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
    """
    Controls the content of the channel properties table

    Parameters:
    ---------------------
    station_id: int
        ID of the station whose properties are displayed

    channel_id: int
        ID of the channel whose properties are displayed
    """
    detector_provider = NuRadioReco.detector.detector_browser.detector_provider.DetectorProvider()
    detector = detector_provider.get_detector()
    if station_id is None or channel_id is None:
        return ''
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
