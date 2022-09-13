import json
import plotly.subplots
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from NuRadioReco.eventbrowser.app import app
from NuRadioReco.utilities import units
from NuRadioReco.eventbrowser.default_layout import default_layout
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.eventbrowser.dataprovider

provider = NuRadioReco.eventbrowser.dataprovider.DataProvider()

layout = [
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
]

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


@app.callback(Output('cr-xcorrelation-dropdown', 'options'),
              [Input('xcorrelation-event-type', 'value')])
def set_xcorrelation_options(event_type):
    if event_type == 'nu':
        return nu_xcorr_options
    else:
        return cr_xcorr_options


@app.callback(Output('cr-xcorrelation', 'figure'),
              [Input('cr-xcorrelation-dropdown', 'value'),
               Input('filename', 'value'),
               Input('event-ids', 'children'),
               Input('station-id-dropdown', 'value'),
               Input('xcorrelation-event-type', 'value')],
              [State('user_id', 'children')])
def plot_corr(xcorr_type, filename, jcurrent_selection, station_id, event_type, juser_id):
    if filename is None or station_id is None or xcorr_type is None:
        return {}
    user_id = json.loads(juser_id)
    nurio = provider.get_file_handler(user_id, filename)
    fig = plotly.subplots.make_subplots(rows=1, cols=1)
    keys = nurio.get_header()[station_id].keys()
    if event_type == 'nu':
        if stnp.nu_xcorrelations not in keys:
            return {}
        xcorrs = nurio.get_header()[station_id][stnp.nu_xcorrelations]
    else:
        if stnp.cr_xcorrelations not in keys:
            return {}
        xcorrs = nurio.get_header()[station_id][stnp.cr_xcorrelations]
    if stnp.station_time in keys:
        times = []
        for time in nurio.get_header()[station_id][stnp.station_time]:
            times.append(time.value)
        current_selection = json.loads(jcurrent_selection)
        fig.append_trace(plotly.graph_objs.Scatter(
            x=times,
            y=[xcorrs[i][xcorr_type] for i in range(len(xcorrs))],
            text=[str(x) for x in nurio.get_event_ids()],
            customdata=[x for x in range(nurio.get_n_events())],
            mode='markers',
            opacity=1,
            selectedpoints=current_selection
        ), 1, 1)
    else:
        return {}

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
def plot_corr_amplitude(xcorr_type, filename, jcurrent_selection, event_type, station_id, juser_id):
    if filename is None or station_id is None or xcorr_type is None:
        return {}
    user_id = json.loads(juser_id)
    nurio = provider.get_file_handler(user_id, filename)
    fig = plotly.subplots.make_subplots(rows=1, cols=1)
    keys = nurio.get_header()[station_id].keys()
    if event_type == 'nu':
        if stnp.nu_xcorrelations not in keys:
            return {}
        xcorrs = nurio.get_header()[station_id][stnp.nu_xcorrelations]
    else:
        if stnp.cr_xcorrelations not in keys:
            return {}
        xcorrs = nurio.get_header()[station_id][stnp.cr_xcorrelations]
    if stnp.channels_max_amplitude in keys:
        current_selection = json.loads(jcurrent_selection)
        fig.append_trace(plotly.graph_objs.Scatter(
            x=nurio.get_header()[station_id][stnp.channels_max_amplitude] / units.mV,
            y=[xcorrs[i][xcorr_type] for i in range(len(xcorrs))],
            text=[str(x) for x in nurio.get_event_ids()],
            customdata=[x for x in range(nurio.get_n_events())],
            mode='markers',
            opacity=1,
            selectedpoints=current_selection
        ), 1, 1)
    else:
        return {}
    fig['layout'].update(default_layout)
    fig['layout']['xaxis'].update({'type': 'log', 'title': 'maximum amplitude [mV]'})
    fig['layout']['yaxis'].update({'title': xcorr_type, 'range': [0, 1]})
    fig['layout']['hovermode'] = 'closest'
    return fig
