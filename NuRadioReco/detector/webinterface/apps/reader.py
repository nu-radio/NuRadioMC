import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from NuRadioReco.utilities import units
from plotly import subplots
import numpy as np
import plotly.graph_objs as go
import json
import sys
import base64
from io import StringIO

from NuRadioReco.detector.detector_mongo import det
# from NuRadioReco.detector.detector_mongo import Detector
# from NuRadioReco.detector.webinterface import config
#from NuRadioReco.detector.webinterface.utils.sparameter_helper import validate_Sdata,  enable_board_name_input, plot_Sparameters, sparameters_layout
from NuRadioReco.detector.webinterface.utils.Vpol_helper import validate_Sdata, plot_Sparameters, sparameters_layout
from NuRadioReco.detector.webinterface.utils.table import get_table
from NuRadioReco.detector.webinterface.utils.units import str_to_unit
from NuRadioReco.detector.webinterface.app import app


table_name = "Vpol"
component_name = "exactvpol"
full_tables = "full_tables"
layout = html.Div([
    html.H3('Read Database Values', id='trigger'),
    dcc.Link('Go back to menu', href='/apps/menu'),
    html.Br(),
    html.Div([html.Div("Select the Catagory and individual component:", style={'float':'left'}),
        dcc.Dropdown(
            id='Table_name_list',
            options=[
                {'label': 'Downhole Cables', 'value': "CABLE"},
                {'label': 'Surface Cables', 'value': "surf_CABLE"},
                {'label': 'Vpol', 'value': "Vpol"},
                {'label': 'Hpol', 'value': "Hpol"},
                {'label': 'DRAB', 'value': "DRAB"},
                {'label': 'IGLU', 'value': "IGLU"},
                {'label': 'DAQ', 'value': "DAQ"},
                {'label': 'DRAB', 'value': "DRAB"},
                ],
            placeholder='Pick Table',
            style={'width': '200px', 'float':'left'}),
        dcc.Dropdown(
            id='Table_name_list',
            options=[
                {'label': 'A54321', 'value': "CABLE"},
                ],
            placeholder='Pick Component',
            style={'width': '200px', 'float':'left'}),

        ]),

    html.Br(),
    html.Br(),
    html.Div([
        html.Button('Export Individual Calibration Value', id=table_name + component_name, disabled=True),
    ], style={'width': "100%", "overflow": "hidden"}),
        html.Div([
            html.Button('Export Full 2021 Stations', id=full_tables, disabled=True),
        ], style={'width': "100%", "overflow": "hidden"}),



    ])



def update_dropdown_VPol_names(n_intervals, options, table_name):
    """
    updates the dropdown menu with existing antenna names from the database
    """
    if(get_table(table_name) is not None):
        for VPol_name in get_table(table_name).distinct("name"):
            options.append(
                {"label": VPol_name, "value": VPol_name}
            )
        return options
