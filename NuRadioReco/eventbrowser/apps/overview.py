from __future__ import absolute_import, division, print_function  # , unicode_literals
from dash.dependencies import Input, Output, State
import time
import dash
import radiotools.helper as hp
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import plotly
from plotly import tools
import json
from app import app
import dataprovider
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.eventbrowser.apps.common import get_point_index
import numpy as np
import logging
logger = logging.getLogger('overview')

provider = dataprovider.DataProvider()


cr_xcorr_options = [
    {'label': 'maximum cr x-corr all channels', 'value': 'cr_max_xcorr'},
    {'label': 'maximum of avg cr x-corr in parallel cr channels', 'value': 'cr_avg_xcorr_parallel_crchannels'},
    {'label': 'maximum cr x-corr cr channels', 'value': 'cr_max_xcorr_crchannels'},
    {'label': 'average cr x-corr cr channels', 'value': 'cr_avg_xcorr_crchannels'},
]
nu_xcorr_options = [
    {'label': 'maximum nu x-corr all channels', 'value': 'nu_max_xcorr'},
    {'label': 'maximum of avg nu x-corr in parallel nu channels', 'value': 'nu_avg_xcorr_parallel_nuchannels'},
    {'label': 'maximum nu x-corr nu channels', 'value': 'nu_max_xcorr_nuchannels'},
    {'label': 'average nu x-corr nu channels', 'value': 'nu_avg_xcorr_nuchannels'}
]
    
layout = html.Div([
    html.Div([
        html.Div([
            html.Div('Station', className='panel-heading'),
            html.Div([
                dcc.RadioItems(
                    id='station-overview-rec-sim',
                    options=[
                        {'label': 'Reconstruction', 'value': 'rec'},
                        {'label': 'Simulation', 'value': 'sim'}
                    ],
                    value='rec'
                ),
                html.Div(id='station-overview-properties')
            ], className='panel-body')
        ], className='panel panel-default', style={'flex':'1'}),
        html.Div([
            html.Div('Channels', className='panel-heading'),
            html.Div([
                dcc.Dropdown(id='dropdown-overview-channels',
                    options=[],
                    multi=True,
                    value=[]
                ),
                html.Div(id='channel-overview-properties')
            ], className='panel-body')
        ], className='panel panel-default', style={'flex':'1'}),
        html.Div([
            html.Div('Electric Fields', className='panel-heading')
        ], className='panel panel-default', style={'flex':'1'})
    ], style={'display': 'flex'}),
    html.Div([
        html.Div('Correlations', className='panel-heading'),
        html.Div([
            html.Div([
                html.Div([
                    dcc.RadioItems(
                        id='xcorrelation-event-type',
                        options=[
                            {'label': 'Neutrino', 'value': 'nu'},
                            {'label': 'Cosmic Ray', 'value': 'cr'}
                        ],
                        value='nu'
                    )
                ], style={'flex': 'none', 'padding-right': '20px'}),
                html.Div([
                    dcc.Dropdown(
                        id='cr-xcorrelation-dropdown',
                        options=[]
                    )
                ], style={'flex': '1'})
            ], style={'display': 'flex'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='cr-xcorrelation'),
                    html.Div(children=json.dumps(None), id='cr-xcorrelation-point-click', style={'display': 'none'})
                ], style={'flex': '1'}),
                html.Div([
                    dcc.Graph(id='cr-xcorrelation-amplitude'),
                    html.Div(children=json.dumps(None), id='cr-xcorrelation-amplitude-point-click', style={'display': 'none'})
                ], style={'flex': '1'})
            ], style={'display': 'flex'})
        ], className='panel-body')
    ], className = 'panel panel-default'),
    html.Div([
        html.Div('Template Time Fit', className='panel-heading'),
        html.Div([
            html.Div(id='trigger', style={'display': 'none'},
                     children=json.dumps(None)),
            html.Div([
                html.Div([
                    html.H5("template time fit")
                ], className="six columns"),
                html.Div([
                    html.H5("cross correlation fitter"),
                    dcc.Graph(id='skyplot-xcorr')
                ], className="six columns")
            ], className='row'),
            html.Div(id='output')
        ], className='panel-body')
    ], className='panel panel-default')
])


@app.callback(Output('cr-xcorrelation-dropdown', 'options'),
            [Input('xcorrelation-event-type', 'value')])
def set_xcorrelation_options(event_type):
    if event_type == 'nu':
        return nu_xcorr_options
    else:
        return cr_xcorr_options


station_properties_for_overview = [
    {
        'label': 'Zenith [deg]',
        'param': stnp.zenith,
        'unit': units.deg
    },{
        'label': 'Azimuth [deg]',
        'param': stnp.azimuth,
        'unit': units.deg
    },{
        'label': 'Neutrino Energy [eV]',
        'param': stnp.nu_energy,
        'unit': units.eV
    },{
        'label': 'Cosmic Ray Energy [eV]',
        'param': stnp.cr_energy,
        'unit': units.eV
    }
]
@app.callback(Output('station-overview-properties', 'children'),
                [Input('filename', 'value'),
                Input('event-counter-slider', 'value'),
                Input('station-id-dropdown', 'value'),
                Input('station-overview-rec-sim', 'value')],
                [State('user_id', 'children')])
def station_overview_properties(filename, evt_counter, station_id, rec_or_sim, juser_id):
    if filename is None or station_id is None:
        return ''
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    if station is None:
        return []
    if rec_or_sim == 'rec':
        prop_station = station
    else:
        if station.get_sim_station() is None:
            return []
        else:
            prop_station = station.get_sim_station()
    if prop_station.is_neutrino():
        event_type = 'Neutrino'
    elif prop_station.is_cosmic_ray():
        event_type = 'Cosmic Ray'
    else:
        event_type = 'Unknown'
    reply = []
    reply.append(
        html.Div([
            html.Div('Event Type:', className='custom-table-td'),
            html.Div(str(event_type), className='custom-table-td custom-table-td-last')
        ], className='custom-table-row')
    )
    for display_prop in station_properties_for_overview:
        if prop_station.has_parameter(display_prop['param']):
            if display_prop['unit'] is not None:
                v = prop_station.get_parameter(display_prop['param'])/display_prop['unit']
            else:
                v = prop_station.get_parameter(display_prop['param'])
            reply.append(
            html.Div([
                html.Div(display_prop['label'], className='custom-table-td'),
                html.Div('{:.2f}'.format(v), className='custom-table-td custom-table-td-last')
            ], className='custom-table-row')
            )
    return reply


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
        reply.append(
            html.Div('Channel {}'.format(channel.get_id()),className='custom-table-td')
        )
    return reply


@app.callback(Output('cr-xcorrelation', 'figure'),
              [Input('cr-xcorrelation-dropdown', 'value'),
               Input('filename', 'value'),
               Input('event-ids', 'children'),
               Input('station-id-dropdown', 'value'),
               Input('xcorrelation-event-type', 'value')],
              [State('user_id', 'children')])
def plot_cr_xcorr(xcorr_type, filename, jcurrent_selection, station_id, event_type, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    traces = []
    keys = ariio.get_header()[station_id].keys()
    if event_type == 'nu':
        if not stnp.nu_xcorrelations in keys:
            return {}
        xcorrs = ariio.get_header()[station_id][stnp.nu_xcorrelations]
    else:
        if not stnp.cr_xcorrelations in keys:
            return {}
        xcorrs = ariio.get_header()[station_id][stnp.cr_xcorrelations]
    if stnp.station_time in keys:
        traces.append(go.Scatter(
            x=ariio.get_header()[station_id][stnp.station_time],
            y=[xcorrs[i][xcorr_type] for i in range(len(xcorrs))],
            text=[str(x) for x in ariio.get_event_ids()],
            customdata=[x for x in range(ariio.get_n_events())],
            mode='markers',
            opacity=1
        ))
    else:
        return {}
    current_selection = json.loads(jcurrent_selection)
    if current_selection != []:
        for trace in traces:
            trace['selectedpoints'] = current_selection
    return {
        'data': traces,
        'layout': go.Layout(
            yaxis={'title': xcorr_type, 'range': [0, 1]},
            hovermode='closest'
        )
    }

@app.callback(Output('cr-xcorrelation-point-click', 'children'),
                [Input('cr-xcorrelation', 'clickData')])
def handle_cr_xcorrelation_point_click(click_data):
    if click_data is None:
        return json.dumps(None)
    event_i = click_data['points'][0]['customdata']
    return json.dumps({
        'event_i': event_i,
        'time': time.time()
    })

@app.callback(Output('cr-xcorrelation-amplitude', 'figure'),
              [Input('cr-xcorrelation-dropdown', 'value'),
               Input('filename', 'value'),
               Input('event-ids', 'children'),
               Input('xcorrelation-event-type', 'value'),
               Input('station-id-dropdown', 'value')],
              [State('user_id', 'children')])
def plot_cr_xcorr_amplitude(xcorr_type, filename, jcurrent_selection, event_type, station_id, juser_id):
    if filename is None or station_id is None or xcorr_type is None:
        return {}
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    traces = []
    keys = ariio.get_header()[station_id].keys()
    if event_type == 'nu':
        if not stnp.nu_xcorrelations in keys:
            return {}
        xcorrs = ariio.get_header()[station_id][stnp.nu_xcorrelations]
    else:
        if not stnp.cr_xcorrelations in keys:
            return {}
        xcorrs = ariio.get_header()[station_id][stnp.cr_xcorrelations]
    if stnp.channels_max_amplitude in keys:
        traces.append(go.Scatter(
            x=ariio.get_header()[station_id][stnp.channels_max_amplitude] / units.mV,
            y=[xcorrs[i][xcorr_type] for i in range(len(xcorrs))],
            text=[str(x) for x in ariio.get_event_ids()],
            customdata=[x for x in range(ariio.get_n_events())],
            mode='markers',
            opacity=1
        ))
    else:
        return {}
    # update with current selection
    current_selection = json.loads(jcurrent_selection)
    if current_selection != []:
        for trace in traces:
            trace['selectedpoints'] = current_selection

    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'type': 'log', 'title': 'maximum amplitude [mV]'},
            yaxis={'title': xcorr_type, 'range': [0, 1]},
            hovermode='closest'
        )
    }

@app.callback(Output('cr-xcorrelation-amplitude-point-click', 'children'),
                [Input('cr-xcorrelation-amplitude', 'clickData')])
def handle_cr_xcorrelation_amplitude_point_click(click_data):
    if click_data is None:
        return json.dumps(None)
    event_i = click_data['points'][0]['customdata']
    return json.dumps({
        'event_i': event_i,
        'time': time.time()
    })
