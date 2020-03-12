import dash_core_components as dcc
import dash_html_components as html

layout = html.Div([
    html.H3('Welcome to NuRadioReco database web interface'),
    dcc.Link('Add S parameter measurement of surface board', href='/apps/add_surface_board'),
    html.Br(),
    dcc.Link('Add another DRAB unit measurement', href='/apps/add_DRAB')
])

