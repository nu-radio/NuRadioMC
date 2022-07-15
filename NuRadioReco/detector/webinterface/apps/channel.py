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


table_name = "station"

layout = html.Div([
    html.H3('Insert Stations and Channels', id='trigger'),
    html.Div(table_name, id='table-name'),
    dcc.Link('Go back to menu', href='/apps/menu'),
    html.Div([html.Div(dcc.Link('Add another Station or Channel', href='/apps/channel', refresh=True), id=table_name + "-menu"),
              html.Div([
    html.H3('', id=table_name + 'override-warning', style={"color": "Red"}),
    html.Div([
    dcc.Checklist(
        id="allow-override",
        options=[
            {'label': 'Allow override of existing entries', 'value': 1}
        ],
        value=[])
    ], style={'width':'20%', 'float': 'hidden'}),
    html.Br(),
    html.Br(),
    html.Div([html.Div("Pick a Station or enter new:", style={'float':'left'}),
        dcc.Dropdown(
            id='station_list',
            options=[
              {'label': 'new Station', 'value': 'new'},
              {'label': '11 (Nanoq)', 'value': 11},
              {'label': '21 (Amaroq)', 'value': 21},
              {'label': '22 (Avinngaq)', 'value': 22},
              ],
            placeholder='Pick your station',
            style={'width': '200px', 'float':'left'}),
        dcc.Input(id="new-station-input",
                  disabled=False,
                  placeholder='new unique Station ID',
                  style={'width': '200px',
                         'float': 'left'}),
        dcc.Input(id="new-station-name",
                  disabled=False,
                  placeholder='new unique Station Name',
                  style={'width': '200px',
                         'float': 'left'}),
    ], style={'width':'100%', 'float': 'hidden'}),
    html.Br(),
    html.Br(),
    html.Div("Enter the UTM GPS coordinates"),
    dcc.Input(id="gpsZone",
              type="number",
              placeholder="23M",
              style={'width': '200px',
                     'float': 'left'}),
    dcc.Input(id="gpsEasting",
              type="number",
              placeholder="xxxxx mE",
              style={'width': '200px',
                     'float': 'left'}),
    dcc.Input(id="gpsNorthing",
              type="number",
              placeholder="xxxxx mN",
              style={'width': '200px',
                     'float': 'left'}),
    html.Br(),
    html.Br(),
    html.H4('', id=table_name + '-validation-global-output'),
    html.Div("false", id='validation-global', style={'display': 'none'}),
    html.Div([
        html.Button('insert Station to DB', id=table_name + '-button-insert', disabled=True),
    ], style={'width':"100%", "overflow": "hidden"}),
    html.Br(),
    html.Br(),
    html.Div([html.Div("Enter Channel information:", style={'float':'left'}),
        dcc.Dropdown(
            id='antenna_list',
            options=[
                {'label': 'LPDA', 'value': "LPDA"},
                {'label': 'VPol', 'value': "VPol"},
                {'label': 'HPol', 'value': "HPol"},
                {'label': 'Pulser', 'value': "Pulser"}
                ],
            placeholder='Pick your Antenna',
            style={'width': '200px', 'float':'left'}),
        dcc.Dropdown(
            id='IGLU_list0',
            options=[
                {'label': 'IGLU', 'value': "IGLU"},
                {'label': 'SURFACE', 'value': "SURFACE"}
                ],
            placeholder='Pick your Amplifier',
            style={'width': '200px', 'float':'left'}),
        dcc.Dropdown(
            id='CABLE_list0',
            options=[
                {'label': 'A54321', 'value': "CABLE"},
                ],
            placeholder='Pick your Cable',
            style={'width': '200px', 'float':'left'}),
        ]),
    html.Br(),
    html.Br(),
    html.Div([html.Div("Enter Positions:", style={'float':'left'}),
        dcc.Input(id="orientation_phi0",
                  type="number",
                  placeholder="orientation_phi",
                  style={'width': '200px',
                         'float': 'left'}),
        dcc.Input(id="orientation_theta0",
                  type="number",
                  placeholder="orientation_theta",
                  style={'width': '200px',
                         'float': 'left'}),
        dcc.Input(id="position_x0",
                  type="number",
                  placeholder="positiion_x",
                  style={'width': '200px',
                         'float': 'left'}),
        dcc.Input(id="position_y0",
                  type="number",
                  placeholder="position_y",
                  style={'width': '200px',
                         'float': 'left'}),
        dcc.Input(id="position_z0",
                  type="number",
                  placeholder="position_z",
                  style={'width': '200px',
                         'float': 'left'}),
        dcc.Input(id="rotation_phi0",
                  type="number",
                  placeholder="rotation_phi",
                  style={'width': '200px',
                         'float': 'left'}),
        dcc.Input(id="rotation_theta0",
                  type="number",
                  placeholder="orientation_theta",
                  style={'width': '200px',
                         'float': 'left'}),
        dcc.Input(id="comission_time0",
                  type="number",
                  placeholder="comission_time",
                  style={'width': '200px',
                         'float': 'left'}),
        dcc.Input(id="decomission_time0",
                  type="number",
                  placeholder="decomission_time",
                  style={'width': '200px',
                         'float': 'left'}),
        ]),
    html.Div([
        html.Button('insert Channel to Station', id=table_name + '-channel-insert', disabled=True),
    ], style={'width':"100%", "overflow": "hidden"}),
    html.Div(id='dd-output-container')
    ], id=table_name + "-main")])])

@app.callback([Output(table_name + '-main', 'style'),
               Output(table_name + '-menu', 'style')],
              [Input(table_name + '-button-insert', 'n_clicks')],
              [State('VPol-list', 'value'),
               State('new-VPol-input', 'value'),
             State('Sdata', 'contents'),
             State('dropdown-frequencies', 'value'),
             State('dropdown-magnitude', 'value'),
             State('separator', 'value'),
             State("function-test", "value"),
             State("protocol", "value")])
def insert_to_db(n_clicks, VPol_dropdown, new_VPol_name, contents, unit_ff, unit_mag, sep, function_test, protocol):
    print(f"n_clicks is {n_clicks}")
    if(not n_clicks is None):
        print("insert to db")
        VPol_name = VPol_dropdown
        if(VPol_dropdown == "new"):
            VPol_name = new_VPol_name
        if('working' not in function_test):
            print(VPol_name)
            det.VPol_set_not_working(VPol_name)
        else:
            content_type, content_string = contents.split(',')
            S_datas = base64.b64decode(content_string)
            S_data_io = StringIO(S_datas.decode('utf-8'))
            S_data = np.genfromtxt(S_data_io, skip_header=17, skip_footer=1, delimiter=sep).T
            S_data[0] *= str_to_unit[unit_ff]
            S_data[1] *= str_to_unit[unit_mag]
            if('primary' not in function_test):
                primary_measurement = False
            else:
                primary_measurement = True
            print(VPol_name, S_data)
            det.VPol_add_Sparameters(VPol_name, S_data, primary_measurement, protocol)

        return {'display': 'none'}, {}
    else:
        return {}, {'display': 'none'}
