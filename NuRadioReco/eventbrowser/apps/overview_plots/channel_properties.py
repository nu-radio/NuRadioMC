import dash
import json
import plotly
import numpy as np
from app import app
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from NuRadioReco.utilities import units
from NuRadioReco.eventbrowser.default_layout import default_layout
from NuRadioReco.framework.parameters import channelParameters as chp
import NuRadioReco.eventbrowser.apps.common
import dataprovider
provider = dataprovider.DataProvider()

layout = [
    dcc.Dropdown(id='dropdown-overview-channels',
        options=[],
        multi=True,
        value=[]
    ),
    html.Div(id='channel-overview-properties')
]

@app.callback(Output('dropdown-overview-channels', 'options'),
                [Input('filename', 'value'),
                Input('event-counter-slider', 'value'),
                Input('station-id-dropdown', 'value'),
                Input('station-overview-rec-sim', 'value')],
                [State('user_id', 'children')])
def dropdown_overview_channels(filename, evt_counter, station_id, rec_or_sim, juser_id):
    if filename is None or station_id is None:
        return ''
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    if station is None:
        return []
    options = []
    for channel in station.iter_channels():
        options.append({
            'label': 'Ch. {}'.format(channel.get_id()),
            'value': channel.get_id()
        })
    return options

channel_properties_for_overview = [
    {
        'label': 'Signal to Noise ratio',
        'param': chp.SNR,
        'unit': None
    },{
        'label': 'Max. amplitude [microVolt]',
        'param': chp.maximum_amplitude,
        'unit': units.microvolt
    },{
        'label': 'Max. of Hilbert envelope [microVolt]',
        'param': chp.maximum_amplitude_envelope,
        'unit': units.microvolt
    },{
        'label': 'Cosmic Ray Template Correlations',
        'param': chp.cr_xcorrelations,
        'unit': None
    },{
        'label': 'Neutrino Template Correlations',
        'param': chp.nu_xcorrelations,
        'unit': None
    }
]

@app.callback(Output('channel-overview-properties', 'children'),
                [Input('filename', 'value'),
                Input('event-counter-slider', 'value'),
                Input('station-id-dropdown', 'value'),
                Input('dropdown-overview-channels', 'value')],
                [State('user_id', 'children')])
def channel_overview_properties(filename, evt_counter, station_id, selected_channels, juser_id):
    if filename is None or station_id is None:
        return ''
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    if station is None:
        return []
    reply = []
    for channel_id in selected_channels:
        channel = station.get_channel(channel_id)
        props = NuRadioReco.eventbrowser.apps.common.get_properties_divs(channel, channel_properties_for_overview)
        reply.append(
            html.Div([
            html.Div([
                html.Div('Channel {}'.format(channel.get_id()), className='custom-table-th')
            ],className='custom-table-row'),
            html.Div(props)
        ]))
    return reply
