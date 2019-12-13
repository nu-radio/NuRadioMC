import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

layout = html.Div([
    html.H3('Welcome to NuRadioReco database web interface'),
    dcc.Link('Add a new amplifier', href='/apps/add_amps')
])

