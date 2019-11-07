from __future__ import absolute_import, division, print_function  # , unicode_literals
import dash_html_components as html
import plotly
from NuRadioReco.utilities import units
import NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_trace
import NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_spectrum
import NuRadioReco.eventbrowser.apps.trace_plots.channel_time_trace
import NuRadioReco.eventbrowser.apps.trace_plots.channel_spectrum
import NuRadioReco.eventbrowser.apps.trace_plots.multi_channel_plot
import numpy as np
import logging

logger = logging.getLogger('traces')

layout = html.Div([
    html.Div(id='trigger-trace', style={'display': 'none'}),
    html.Div([
        html.Div([
            html.Div('Electric Field Traces', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_trace.layout,
            className='panel-body')
        ], className='panel panel-default', style={'flex': '1'}),
        html.Div([
            html.Div('Electric Field Spectrum', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_spectrum.layout,
            className='panel-body')
        ], className='panel panel-default', style={'flex': '1'})
    ], style={'display': 'flex'}),
    html.Div([
        html.Div([
            html.Div('Channel Traces', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.trace_plots.channel_time_trace.layout,
            className='panel-body')
        ], className='panel panel-default', style={'flex': '1'}),
        html.Div([
            html.Div('Channel Spectrum', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.trace_plots.channel_spectrum.layout,
            className='panel-body')
        ], className='panel panel-default', style={'flex': '1'})
    ], style={'display': 'flex'}),
    html.Div([
        html.Div([
            html.Div('Individual Channels', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.trace_plots.multi_channel_plot.layout, className='panel-body')
        ], className='panel panel-default', style={'flex': '1'})
    ], style={'display': 'flex'})
])
