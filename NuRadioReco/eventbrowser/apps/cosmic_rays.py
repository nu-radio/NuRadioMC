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
logger = logging.getLogger('traces')

provider = dataprovider.DataProvider()

layout = html.Div([
    #Sim Traces Plot
    html.Div([
        html.Div([
            html.Div([
                html.Div('Polarization', className='panel-heading'),
                html.Div([
                    html.Div([
                        html.Div([
                            dcc.Graph(id='cr-polarization-zenith')
                        ], style={'flex': '1'}),
                    ], style={'display': 'flex'})
                ], className='panel-body')
            ], className='panel panel-default', style={'flex': '1'}),
            html.Div([
                html.Div('Direction Reconstruction', className='panel-heading', style={'display': 'flex'}),
                html.Div([
                    dcc.Graph(id='cr-skyplot'),
                ], className='panel-body')
            ], className='panel panel-default', style={'flex': '1'})
        ], style={'display': 'flex'})
    ])
])


@app.callback(Output('cr-polarization-zenith', 'figure'),
              [Input('filename', 'value'),
               Input('btn-open-file', 'value'),
               Input('event-ids', 'children'),
               Input('station-id-dropdown', 'value')],
              [State('user_id', 'children')])
def plot_cr_polarization_zenith(filename, btn, jcurrent_selection, station_id, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    traces = []
    keys = ariio.get_header()[station_id].keys()
    pol = []
    pol_exp = []
    zeniths = []
    for i_event in range(ariio.get_n_events()):
        event = ariio.get_event_i(i_event)
        for station in event.get_stations():
            for electric_field in station.get_electric_fields():
                if electric_field.has_parameter(efp.polarization_angle) and electric_field.has_parameter(efp.polarization_angle_expectation) and electric_field.has_parameter(efp.zenith):
                    pol.append(electric_field.get_parameter(efp.polarization_angle))
                    pol_exp.append(electric_field.get_parameter(efp.polarization_angle_expectation))
                    zeniths.append(electric_field.get_parameter(efp.zenith))
    pol = np.array(pol)
    pol = np.abs(pol)
    pol[pol > 0.5 * np.pi] = np.pi - pol[pol > 0.5 * np.pi]
    pol_exp = np.array(pol_exp)
    pol_exp = np.abs(pol_exp)
    pol_exp[pol_exp > 0.5 * np.pi] = np.pi - pol_exp[pol_exp > 0.5 * np.pi]
    zeniths = np.array(zeniths)
    traces.append(go.Scatter(
        x=zeniths / units.deg,
        y=np.abs(pol - pol_exp) / units.deg,
        text=[str(x) for x in ariio.get_event_ids()],
        mode='markers',
        customdata=[x for x in range(ariio.get_n_events())],
        opacity=1
    ))

    # update with current selection
    current_selection = json.loads(jcurrent_selection)
    if current_selection != []:
        for trace in traces:
            trace['selectedpoints'] = current_selection

    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'type': 'linear', 'title': 'zenith angle [deg]'},
            yaxis={'title': 'polarization angle error [deg]', 'range': [0, 90]},
#             margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
#             legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }    

@app.callback(Output('cr-skyplot', 'figure'),
              [Input('filename', 'value'),
               Input('trigger', 'children'),
               Input('event-ids', 'children'),
               Input('btn-open-file', 'value'),
               Input('station-id-dropdown', 'value')],
              [State('user_id', 'children')])
def plot_skyplot(filename, trigger, jcurrent_selection, btn, station_id, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    current_selection = json.loads(jcurrent_selection)
    ariio = provider.get_arianna_io(user_id, filename)
    traces = []
    keys = ariio.get_header()[station_id].keys()
    if stnp.cr_xcorrelations not in ariio.get_header()[station_id]:
        return {}
    xcorrs = ariio.get_header()[station_id][stnp.cr_xcorrelations]
    if stnp.cr_zenith in keys and stnp.cr_azimuth in keys:
        traces.append(go.Scatterpolar(
            r=np.rad2deg(ariio.get_header()[station_id][stnp.cr_zenith]),
            theta=np.rad2deg(ariio.get_header()[station_id][stnp.cr_azimuth]),
            text=[str(x) for x in ariio.get_event_ids()],
            mode='markers',
            name='cosmic ray events',
            opacity=1,
            customdata=[x for x in range(ariio.get_n_events())],
            marker=dict(
                color='blue'
            )
        ))
    # update with current selection
    if current_selection != []:
        for trace in traces:
            trace['selectedpoints'] = current_selection

    return {
        'data': traces,
        'layout': go.Layout(
            showlegend=True,
            hovermode='closest',
            height=500
        )
    }

