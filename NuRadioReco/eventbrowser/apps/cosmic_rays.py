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
    html.Div([
        html.Div([
            html.Div([
                html.Div('Polarization', className='panel-heading'),
                html.Div(NuRadioReco.eventbrowser.apps.cosmic_ray_plots.cosmic_ray_polarization_zenith.layout, className='panel-body')
            ], className='panel panel-default', style={'flex': '1'}),
            html.Div([
                html.Div('Direction Reconstruction', className='panel-heading', style={'display': 'flex'}),
                html.Div(NuRadioReco.eventbrowser.apps.cosmic_ray_plots.cosmic_ray_skyplot.layout, className='panel-body')
            ], className='panel panel-default', style={'flex': '1'})
        ], style={'display': 'flex'})
    ])
])
