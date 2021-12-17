from __future__ import absolute_import, division, print_function  # , unicode_literals
from dash import html
import NuRadioReco.eventbrowser.apps.simulation_plots.sim_electric_field_trace
import NuRadioReco.eventbrowser.apps.simulation_plots.sim_electric_field_spectrum
import NuRadioReco.eventbrowser.apps.simulation_plots.sim_event_overview
import logging

logger = logging.getLogger('traces')
parent_logger = logging.getLogger('NuRadioReco')
logger.setLevel(parent_logger.level)

layout = html.Div([
    html.Div([
        html.Div([
            html.Div('Sim Traces', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.simulation_plots.sim_electric_field_trace.layout,
                     className='panel-body', style={'min-height': '500px'})
        ], className='panel panel-default mb-2', style={'flex': '1'}),
        html.Div([
            html.Div('Sim Spectrum', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.simulation_plots.sim_electric_field_spectrum.layout,
                     className='panel-body', style={'min-height': '500px'})
        ], className='panel panel-default mb-2', style={'flex': '1'})
    ], style={'display': 'flex'}),
    html.Div([
        html.Div([
            html.Div('Simulated Event', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.simulation_plots.sim_event_overview.layout,
                     className='panel-body', style={'display': 'flex'})
        ], className='panel panel-default', style={'flex': '1'})
    ], style={'display': 'flex'})
])
