from __future__ import absolute_import, division, print_function  # , unicode_literals
from dash import html
import NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_trace
import NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_spectrum
import NuRadioReco.eventbrowser.apps.trace_plots.channel_time_trace
import NuRadioReco.eventbrowser.apps.trace_plots.channel_spectrum
import NuRadioReco.eventbrowser.apps.trace_plots.multi_channel_plot
from dash.dependencies import State, Input, Output
from NuRadioReco.eventbrowser.app import app
import logging

logger = logging.getLogger('traces')
parent_logger = logging.getLogger('NuRadioReco')
logger.setLevel(parent_logger.level)

layout = html.Div([
    html.Div(id='trigger-trace', style={'display': 'none'}),
    html.Div([
        html.Div([
            html.Div([
                'Electric Field Traces',
                html.Button('Show', id='toggle_efield_traces', n_clicks=0, style={'float':'right'})
            ], className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_trace.layout,
                     className='panel-body', id='efield_traces_layout')
        ], className='panel panel-default', style={'flex': '1'}),
        html.Div([
            html.Div([
                'Electric Field Spectrum',
                html.Button('Show', id='toggle_efield_spectrum', n_clicks=0, style={'float':'right'})
            ], className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_spectrum.layout,
                     className='panel-body', id='efield_spectrum_layout')
        ], className='panel panel-default', style={'flex': '1'})
    ], style={'display': 'flex'}),
    html.Div([
        html.Div([
            html.Div([
                'Channel Traces',
                html.Button('Show', id='toggle_channel_traces', n_clicks=0, style={'float':'right'})
                ], className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.trace_plots.channel_time_trace.layout,
                     className='panel-body', id='channel_traces_layout')
        ], className='panel panel-default', style={'flex': '1'}),
        html.Div([
            html.Div([
                'Channel Spectrum', 
                html.Button('Show', id='toggle_channel_spectrum', n_clicks=0, style={'float':'right'})
                ], className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.trace_plots.channel_spectrum.layout,
                     className='panel-body', id='channel_spectrum_layout')
        ], className='panel panel-default', style={'flex': '1'})
    ], style={'display': 'flex'}),
    html.Div([
        html.Div([
            html.Div('Individual Channels', className='panel-heading'),
            html.Div(NuRadioReco.eventbrowser.apps.trace_plots.multi_channel_plot.layout, className='panel-body')
        ], className='panel panel-default', style={'flex': '1'})
    ], style={'display': 'flex'})
])

@app.callback(
    [Output('channel_traces_layout', 'children'),
    Output('toggle_channel_traces', 'children')],
    [Input('toggle_channel_traces', 'n_clicks')],
    State('toggle_channel_traces', 'children'),
    prevent_initial_callbacks=True
)
def toggle_channel_trace_plot(button_clicks, showhide):
    if showhide == 'Hide':
        return [], 'Show'
    else:
        return NuRadioReco.eventbrowser.apps.trace_plots.channel_time_trace.layout, 'Hide'

@app.callback(
    [Output('channel_spectrum_layout', 'children'),
    Output('toggle_channel_spectrum', 'children')],
    [Input('toggle_channel_spectrum', 'n_clicks')],
    State('toggle_channel_spectrum', 'children'),
    prevent_initial_callbacks=True
)
def toggle_channel_spectrum_plot(button_clicks, showhide):
    if showhide == 'Hide':
        return [], 'Show'
    else:
        return NuRadioReco.eventbrowser.apps.trace_plots.channel_spectrum.layout, 'Hide'

@app.callback(
    [Output('efield_traces_layout', 'children'),
    Output('toggle_efield_traces', 'children')],
    [Input('toggle_efield_traces', 'n_clicks')],
    State('toggle_efield_traces', 'children'),
    prevent_initial_callbacks=True
)
def toggle_efield_traces_plot(button_clicks, showhide):
    if showhide == 'Hide':
        return [], 'Show'
    else:
        return NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_trace.layout, 'Hide'

@app.callback(
    [Output('efield_spectrum_layout', 'children'),
    Output('toggle_efield_spectrum', 'children')],
    [Input('toggle_efield_spectrum', 'n_clicks')],
    State('toggle_efield_spectrum', 'children'),
    prevent_initial_callbacks=True
)
def toggle_efield_spectrum_plot(button_clicks, showhide):
    if showhide == 'Hide':
        return [], 'Show'
    else:
        return NuRadioReco.eventbrowser.apps.trace_plots.rec_electric_field_spectrum.layout, 'Hide'