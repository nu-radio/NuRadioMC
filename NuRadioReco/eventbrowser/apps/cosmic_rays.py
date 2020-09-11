from __future__ import absolute_import, division, print_function  # , unicode_literals
import dash_html_components as html
import NuRadioReco.eventbrowser.dataprovider
import NuRadioReco.eventbrowser.apps.cosmic_ray_plots.cosmic_ray_skyplot
import NuRadioReco.eventbrowser.apps.cosmic_ray_plots.cosmic_ray_polarization_zenith
import logging
logger = logging.getLogger('traces')

provider = NuRadioReco.eventbrowser.dataprovider.DataProvider()

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
