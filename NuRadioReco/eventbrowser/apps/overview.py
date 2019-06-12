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
from NuRadioReco.eventbrowser.default_layout import default_layout
import numpy as np
import logging
import numbers
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
            html.Div('Electric Fields', className='panel-heading'),
            html.Div([
                dcc.RadioItems(
                    id='efield-overview-rec-sim',
                    options=[
                        {'label': 'Reconstruction', 'value': 'rec'},
                        {'label': 'Simulation', 'value': 'sim'}
                    ],
                    value='rec'
                ),
                html.Div(id='efield-overview-properties')
            ], className='panel-body')
        ], className='panel panel-default', style={'flex':'1'}),
        html.Div([
            html.Div('Triggers', className='panel-heading'),
            html.Div(id='trigger-overview-properties')
        ], className='panel panel-default', style={'flex': '1'})
    ], style={'display': 'flex', 'margin': '20px 0'}),
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
                ], style={'flex': '1'}),
                html.Div([
                    dcc.Graph(id='cr-xcorrelation-amplitude'),
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


def get_properties_divs(obj, props_dic):
    props = []
    for display_prop in props_dic:
        if obj.has_parameter(display_prop['param']):
            if type(obj.get_parameter(display_prop['param'])) is dict:
                dict_entries = []
                dic = obj.get_parameter(display_prop['param'])
                for key in dic:
                    if isinstance(dic[key], numbers.Number):
                        dict_entries.append(
                            html.Div([
                                html.Div(key, className='custom-table-td'),
                                html.Div('{:.2f}'.format(dic[key]), className='custom-table-td custom-table-td-last')
                            ], className='custom-table-row')
                        )
                prop = html.Div(dict_entries, className='custom-table-td')
            else:
                if display_prop['unit'] is not None:
                    v = obj.get_parameter(display_prop['param'])/display_prop['unit']
                else:
                    v = obj.get_parameter(display_prop['param'])
                if isinstance(v,float) or isinstance(v, int):
                    prop = html.Div('{:.2f}'.format(v), className='custom-table-td custom-table-td-last')
                else:
                    prop = html.Div('{}'.format(v), className='custom-table-td custom-table-td-last')
            props.append(html.Div([
                html.Div(display_prop['label'], className='custom-table-td'),
                prop
            ], className='custom-table-row'))
    return props

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
    props = get_properties_divs(prop_station, station_properties_for_overview)
    for prop in props:
        reply.append(prop)
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
        props = get_properties_divs(channel, channel_properties_for_overview)
        
        reply.append(
            html.Div([
            html.Div([
                html.Div('Channel {}'.format(channel.get_id()), className='custom-table-th')
            ],className='custom-table-row'),
            html.Div(props)
        ]))
    return reply

efield_properties_for_overview = [
    {
        'label': 'Ray Path Type',
        'param': efp.ray_path_type,
        'unit': None
    },{
        'label': 'Zenith [deg]',
        'param': efp.zenith,
        'unit': units.deg
    },{
        'label': 'Azimuth [deg]',
        'param': efp.azimuth,
        'unit': units.deg
    },{
        'label': 'spectrum Slope',
        'param': efp.cr_spectrum_slope,
        'unit': None
    },{
        'label': 'Energy Fluence [eV]',
        'param': efp.signal_energy_fluence,
        'unit': units.eV
    },{
        'label': 'Polarization Angle [deg]',
        'param': efp.polarization_angle,
        'unit': units.deg
    },{
        'label': 'Expected Polarization Angle [deg]',
        'param': efp.polarization_angle_expectation,
        'unit': units.deg
    }
]

@app.callback(Output('efield-overview-properties', 'children'),
                [Input('filename', 'value'),
                Input('event-counter-slider', 'value'),
                Input('station-id-dropdown', 'value'),
                Input('efield-overview-rec-sim', 'value')],
                [State('user_id', 'children')])
def efield_overview_properties(filename, evt_counter, station_id, rec_sim, juser_id):
    if filename is None or station_id is None:
        return ''
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    if station is None:
        return []
    reply = []
    if rec_sim == 'rec':
        chosen_station = station
    else:
        if station.get_sim_station() is None:
            return {}
        chosen_station = station.get_sim_station()
    for electric_field in chosen_station.get_electric_fields():
        props = get_properties_divs(electric_field, efield_properties_for_overview)
        reply.append(html.Div([
            html.Div('Channels', className='custom-table-td'),
            html.Div('{}'.format(electric_field.get_channel_ids()), className='custom-table-td custom-table-td-last')
        ], className='custom-table-row'))
        reply.append(html.Div(props, style={'margin': '0 0 30px'}))
    return reply
    

@app.callback(Output('trigger-overview-properties', 'children'),
                [Input('filename', 'value'),
                Input('event-counter-slider', 'value'),
                Input('station-id-dropdown', 'value')],
                [State('user_id', 'children')])
def trigger_overview_properties(filename, evt_counter, station_id, juser_id):
    if filename is None or station_id is None:
        return ''
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    if station is None:
        return []
    reply = []
    for trigger_name in station.get_triggers():
        props = [
        html.Div([
            html.Div('{}'.format(trigger_name), className='custom-table-th')
        ], className='custom-table-row')
        ]
        trigger = station.get_trigger(trigger_name)
        for setting_name in trigger.get_trigger_settings():
            props.append(
                html.Div([
                    html.Div('{}'.format(setting_name), className='custom-table-td'),
                    html.Div('{}'.format(trigger.get_trigger_settings()[setting_name]), className='custom-table-td custom-table-td-last')
                ], className='custom-table-row')
            )
        reply.append(html.Div(props))
    return reply


@app.callback(Output('cr-xcorrelation', 'figure'),
              [Input('cr-xcorrelation-dropdown', 'value'),
               Input('filename', 'value'),
               Input('event-ids', 'children'),
               Input('station-id-dropdown', 'value'),
               Input('xcorrelation-event-type', 'value')],
              [State('user_id', 'children')])
def plot_cr_xcorr(xcorr_type, filename, jcurrent_selection, station_id, event_type, juser_id):
    if filename is None or station_id is None or xcorr_type is None:
        return {}
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    fig = tools.make_subplots(rows=1, cols=1)
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
        fig.append_trace(go.Scatter(
            x=ariio.get_header()[station_id][stnp.station_time],
            y=[xcorrs[i][xcorr_type] for i in range(len(xcorrs))],
            text=[str(x) for x in ariio.get_event_ids()],
            customdata=[x for x in range(ariio.get_n_events())],
            mode='markers',
            opacity=1
        ),1,1)
    else:
        return {}
    current_selection = json.loads(jcurrent_selection)
    if current_selection != []:
        for trace in traces:
            trace['selectedpoints'] = current_selection
    fig['layout'].update(default_layout)
    fig['layout']['yaxis'].update({'title': xcorr_type, 'range': [0, 1]})
    fig['layout']['hovermode'] = 'closest'
    return fig


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
    fig = tools.make_subplots(rows=1, cols=1)
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
        fig.append_trace(go.Scatter(
            x=ariio.get_header()[station_id][stnp.channels_max_amplitude] / units.mV,
            y=[xcorrs[i][xcorr_type] for i in range(len(xcorrs))],
            text=[str(x) for x in ariio.get_event_ids()],
            customdata=[x for x in range(ariio.get_n_events())],
            mode='markers',
            opacity=1
        ),1,1)
    else:
        return {}
    # update with current selection
    current_selection = json.loads(jcurrent_selection)
    if current_selection != []:
        for trace in traces:
            trace['selectedpoints'] = current_selection
    fig['layout'].update(default_layout)
    fig['layout']['xaxis'].update({'type': 'log', 'title': 'maximum amplitude [mV]'})
    fig['layout']['yaxis'].update({'title': xcorr_type, 'range': [0, 1]})
    fig['layout']['hovermode'] = 'closest'
    return fig

