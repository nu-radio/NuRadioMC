from __future__ import absolute_import, division, print_function  # , unicode_literals
import time
import radiotools.helper as hp
import dash_html_components as html
import dataprovider
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.eventbrowser.apps.common import get_point_index
from NuRadioReco.eventbrowser.default_layout import default_layout
import NuRadioReco.eventbrowser.apps.overview_plots.template_correlation
import NuRadioReco.eventbrowser.apps.overview_plots.rec_directions
import NuRadioReco.eventbrowser.apps.overview_plots.station_properties
import NuRadioReco.eventbrowser.apps.overview_plots.channel_properties
import NuRadioReco.eventbrowser.apps.overview_plots.electric_field_properties
import NuRadioReco.eventbrowser.apps.overview_plots.trigger_properties
import NuRadioReco.eventbrowser.apps.overview_plots.event_overview
import numpy as np
import logging
logger = logging.getLogger('overview')

provider = dataprovider.DataProvider()

layout = html.Div([
    html.Div([
        html.Div('Event Overview', className='panel-heading'),
        html.Div(NuRadioReco.eventbrowser.apps.overview_plots.event_overview.layout,
            className='panel-body')
    ], className='panel panel-default'),
    html.Div([
        html.Div([
            html.Div('Station', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.overview_plots.station_properties.layout,
            className='panel-body')
        ], className='panel panel-default', style={'flex':'1'}),
        html.Div([
            html.Div('Channels', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.overview_plots.channel_properties.layout,
            className='panel-body')
        ], className='panel panel-default', style={'flex':'1'}),
        html.Div([
            html.Div('Electric Fields', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.overview_plots.electric_field_properties.layout,
            className='panel-body')
        ], className='panel panel-default', style={'flex':'1'}),
        html.Div([
            html.Div('Triggers', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.overview_plots.trigger_properties.layout,
            className='panel-body')
        ], className='panel panel-default', style={'flex': '1'})
    ], style={'display': 'flex', 'margin': '20px 0'}),
    html.Div([
        html.Div('Correlations', className='panel-heading'),
        html.Div(NuRadioReco.eventbrowser.apps.overview_plots.template_correlation.layout,
        className='panel-body')
    ], className = 'panel panel-default'),
    html.Div([
        html.Div('Reconstructed Directions', className='panel-heading'),
        html.Div(NuRadioReco.eventbrowser.apps.overview_plots.rec_directions.layout,
        className='panel-body')
    ], className='panel panel-default')
])
