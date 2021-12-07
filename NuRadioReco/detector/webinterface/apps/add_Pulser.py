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
import csv

from NuRadioReco.detector import detector_mongo as det
from NuRadioReco.detector.webinterface.utils.sparameter_helper import validate_Sdata, update_dropdown_amp_names, enable_board_name_input, plot_Sparameters, sparameters_layout
from NuRadioReco.detector.webinterface.utils.table import get_table
from NuRadioReco.detector.webinterface.utils.units import str_to_unit
from NuRadioReco.detector.webinterface.app import app

# det = detector_mongo.Detector()

table_name = "PULSER"

layout = html.Div([
    html.H3('Add a measurment of a Pulser', id='trigger'),
    html.Div(table_name, id='table-name'),
    dcc.Link('Go back to menu', href='/apps/menu'),
    ])
