import json
from NuRadioReco.eventbrowser.app import app
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import channelParameters as chp
import NuRadioReco.eventbrowser.apps.common
import NuRadioReco.eventbrowser.dataprovider
import logging

logger = logging.getLogger('overview')
provider = NuRadioReco.eventbrowser.dataprovider.DataProvider()

layout = [
    dcc.RadioItems(
        id='channel-overview-rec-sim',
        options=[
            {'label': 'Reconstruction', 'value': 'rec'},
            {'label': 'Simulation', 'value': 'sim'}
        ],
        value='rec'),
    dcc.Dropdown(id='dropdown-overview-channels',
                 options=[],
                 multi=True,
                 value=[]
                 ),
    html.Div(id='channel-overview-properties')
]


@app.callback(Output('dropdown-overview-channels', 'options'),
              Output('dropdown-overview-channels', 'value'),
              [Input('filename', 'value'),
               Input('event-counter-slider', 'value'),
               Input('station-id-dropdown', 'value')],
               Input('channel-overview-rec-sim', 'value'),
              [State('user_id', 'children')])
def dropdown_overview_channels(filename, evt_counter, station_id, rec_or_sim, juser_id):
    if filename is None or station_id is None:
        return [], []
    user_id = json.loads(juser_id)
    nurio = provider.get_file_handler(user_id, filename)
    evt = nurio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    if rec_or_sim == 'sim':
        if station.has_sim_station():
            station = station.get_sim_station()
        else:
            logger.warning('Selected station has no sim_station')
            return [], []
    if station is None:
        return [], []
    options = []
    for channel in station.iter_channels():
        if rec_or_sim == 'rec':
            options.append({
                'label': 'Ch. {}'.format(channel.get_id()),
                'value': channel.get_id()
            })
        else:
            id_list = (
                channel.get_id(), channel.get_shower_id(), channel.get_ray_tracing_solution_id()
            )
            options.append({
                'label': 'Ch. {} / shower {} / ray {}'.format(*id_list),
                'value': '_'.join([str(k) for k in id_list])
            })
    return sorted(options, key=lambda k: k['value']), []


channel_properties_for_overview = [
    {
        'label': 'Signal to Noise ratio',
        'param': chp.SNR,
        'unit': None
    }, {
        'label': 'Max. amplitude [microVolt]',
        'param': chp.maximum_amplitude,
        'unit': units.microvolt
    }, {
        'label': 'Max. of Hilbert envelope [microVolt]',
        'param': chp.maximum_amplitude_envelope,
        'unit': units.microvolt
    }, {
        'label': 'Cosmic Ray Template Correlations',
        'param': chp.cr_xcorrelations,
        'unit': None
    }, {
        'label': 'Neutrino Template Correlations',
        'param': chp.nu_xcorrelations,
        'unit': None
    }
]


@app.callback(Output('channel-overview-properties', 'children'),
              [Input('filename', 'value'),
               Input('event-counter-slider', 'value'),
               Input('station-id-dropdown', 'value'),
               Input('dropdown-overview-channels', 'value'),
               Input('channel-overview-rec-sim', 'value')],
              [State('user_id', 'children')])
def channel_overview_properties(filename, evt_counter, station_id, selected_channels, rec_or_sim, juser_id):
    if filename is None or station_id is None:
        return []
    user_id = json.loads(juser_id)
    nurio = provider.get_file_handler(user_id, filename)
    evt = nurio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    if station is None:
        return []
    if rec_or_sim == 'sim':
        if station.has_sim_station():
            station = station.get_sim_station()
        else:
            return []
    reply = []
    for channel_id in selected_channels:
        if rec_or_sim == 'rec':
            channel_title = 'Channel {}'.format(channel_id)
        else:
            channel_id = tuple(int(k) for k in channel_id.split('_'))
            channel_title = 'Channel {} / shower {} / ray {}'.format(*channel_id)
        channel = station.get_channel(channel_id)    
        props = NuRadioReco.eventbrowser.apps.common.get_properties_divs(channel, channel_properties_for_overview)
        reply.append(
            html.Div([
                html.Div([
                    html.Div(channel_title, className='custom-table-th')
                ], className='custom-table-row'),
                html.Div(props)
            ]))
    return reply
