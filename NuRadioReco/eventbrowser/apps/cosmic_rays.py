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
import NuRadioReco.eventbrowser.apps.cosmic_ray_plots.cosmic_ray_skyplot
import NuRadioReco.eventbrowser.apps.cosmic_ray_plots.cosmic_ray_polarization_zenith
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
    return NuRadioReco.eventbrowser.apps.cosmic_ray_plots.cosmic_ray_polarization_zenith.plot_cr_polarization_zenith(filename, btn, jcurrent_selection, station_id, juser_id, provider)

@app.callback(Output('cr-skyplot', 'figure'),
              [Input('filename', 'value'),
               Input('trigger', 'children'),
               Input('event-ids', 'children'),
               Input('btn-open-file', 'value'),
               Input('station-id-dropdown', 'value')],
              [State('user_id', 'children')])
def plot_skyplot(filename, trigger, jcurrent_selection, btn, station_id, juser_id):
    return NuRadioReco.eventbrowser.apps.cosmic_ray_plots.cosmic_ray_skyplot.cosmic_ray_skyplot(filename, trigger, jcurrent_selection, btn, station_id, juser_id, provider)
