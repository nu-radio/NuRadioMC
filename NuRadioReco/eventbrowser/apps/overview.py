from __future__ import absolute_import, division, print_function  # , unicode_literals
from dash import html
import NuRadioReco.eventbrowser.dataprovider
import NuRadioReco.eventbrowser.apps.overview_plots.template_correlation
import NuRadioReco.eventbrowser.apps.overview_plots.rec_directions
import NuRadioReco.eventbrowser.apps.overview_plots.station_properties
import NuRadioReco.eventbrowser.apps.overview_plots.channel_properties
import NuRadioReco.eventbrowser.apps.overview_plots.electric_field_properties
import NuRadioReco.eventbrowser.apps.overview_plots.trigger_properties
import NuRadioReco.eventbrowser.apps.overview_plots.event_overview
import logging

logger = logging.getLogger('overview')
parent_logger = logging.getLogger('NuRadioReco')
logger.setLevel(parent_logger.level)

provider = NuRadioReco.eventbrowser.dataprovider.DataProvider()

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
        ], className='panel panel-default', style={'flex': '1'}),
        html.Div([
            html.Div('Channels', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.overview_plots.channel_properties.layout,
                     className='panel-body')
        ], className='panel panel-default', style={'flex': '1'}),
        html.Div([
            html.Div('Electric Fields', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.overview_plots.electric_field_properties.layout,
                     className='panel-body')
        ], className='panel panel-default', style={'flex': '1'}),
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
    ], className='panel panel-default'),
    html.Div([
        html.Div('Reconstructed Directions', className='panel-heading'),
        html.Div(NuRadioReco.eventbrowser.apps.overview_plots.rec_directions.layout,
                 className='panel-body')
    ], className='panel panel-default')
])
