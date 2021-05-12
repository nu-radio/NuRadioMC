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

from NuRadioReco.detector import detector_mongo as det
#from NuRadioReco.detector.webinterface.utils.sparameter_helper import validate_Sdata,  enable_board_name_input, plot_Sparameters, sparameters_layout
from NuRadioReco.detector.webinterface.utils.Vpol_helper import validate_Sdata, plot_Sparameters, sparameters_layout
from NuRadioReco.detector.webinterface.utils.table import get_table
from NuRadioReco.detector.webinterface.utils.units import str_to_unit
from NuRadioReco.detector.webinterface.app import app

table_name = "VPol"

layout = html.Div([
    html.H3('Add S11 measurment of VPol Antenna', id='trigger'),
    html.Div(table_name, id='table-name'),
    dcc.Link('Go back to menu', href='/apps/menu'),
    html.Div([html.Div(dcc.Link('Add another VPol unit measurement', href='/apps/add_VPol', refresh=True), id=table_name + "-menu"),
              html.Div([
    html.H3('', id=table_name + 'override-warning', style={"color": "Red"}),
    html.Div([
    dcc.Checklist(
        id="allow-override",
        options=[
            {'label': 'Allow override of existing entries', 'value': 1}
        ],
        value=[],
        style={'width': '15%'}
    ], style={'width':'100%', 'float': 'hidden'}),
    html.Br(),
    html.Br(),
    html.Div([html.Div("Select existing antenna or enter unique name of new antenna:", style={'float':'left'}),
        dcc.Dropdown(
            id='VPol-list',
            options=[
                {'label': 'new VPol', 'value': "new"}
            ],
            value="new",
            style={'width': '200px', 'float':'left'}
        ),
        dcc.Input(id="new-VPol-input",
                  disabled=False,
                  placeholder='new unique VPol name',
                  style={'width': '200px',
                         'float': 'left'}),
    ], style={'width':'100%', 'float': 'hidden'}),
    html.Br(),
    html.Br(),
    sparameters_layout,
    html.H4('', id=table_name + '-validation-global-output'),
    html.Div("false", id='validation-global', style={'display': 'none'}),
    html.Div([
        html.Button('insert to DB', id=table_name + '-button-insert', disabled=True),
    ], style={'width':"100%", "overflow": "hidden"}),
    html.Div(id='dd-output-container'),
    dcc.Graph(id='figure-VPol', style={"height": "1000px", "width" : "100%"})
    ], id=table_name + "-main")])])


@app.callback(
    [Output('new-VPol-input', 'disabled'),
     Output(table_name + "override-warning", "children")],
    [Input('VPol-list', 'value')])
def enable_VPol_name_input(value):
    """
    enable text field for new VPol unit
    """
    if(value == "new"):
        return False, ""
    else:
        return True, f"You are about to override the VPol unit {value}!"


@app.callback(
    Output("VPol-list", "options"),
    [Input("trigger", "children")],
    [State("VPol-list", "options"),
     State("table-name", "children")]
)
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


@app.callback(
    [
        Output(table_name + "-validation-global-output", "children"),
        Output(table_name + "-validation-global-output", "style"),
        Output(table_name + "-validation-global-output", "data-validated"),
        Output(table_name + '-button-insert', 'disabled')
    ],
    [Input("validation-S11data-output", "data-validated"),
     Input('VPol-list', 'value'),
     Input('new-VPol-input', 'value'),
     Input("function-test", "value")])
def validate_global(Sdata_validated, VPol_dropdown, new_VPol_name, function_test):
    """
    validates all three inputs, this callback is triggered by the individual input validation
    """
    if(VPol_dropdown == ""):
        return "antenna name not set", {"color": "Red"}, False, True
    if(VPol_dropdown == "new" and (new_VPol_name is None or new_VPol_name == "")):
        return "antenna name dropdown set to new but no new antenna name was entered", {"color": "Red"}, False, True

    print(function_test)
    if('working' not in function_test):
        return "all inputs validated", {"color": "Green"}, True, False
    elif(Sdata_validated):
        return "all inputs validated", {"color": "Green"}, True, False

    return "input fields not validated", {"color": "Red"}, False, True


@app.callback([Output(table_name + '-main', 'style'),
               Output(table_name + '-menu', 'style')],
              [Input(table_name + '-button-insert', 'n_clicks')],
              [State('VPol-list', 'value'),
               State('new-VPol-input', 'value'),
             State('Sdata', 'contents'),
             State('dropdown-frequencies', 'value'),
             State('dropdown-magnitude', 'value'),
             State('separator', 'value'),
             State("function-test", "value")])
def insert_to_db(n_clicks, VPol_dropdown, new_VPol_name, contents, unit_ff, unit_mag, sep, function_test):
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
            print(VPol_name, S_data)
            det.VPol_add_Sparameters(VPol_name, S_data)

        return {'display': 'none'}, {}
    else:
        return {}, {'display': 'none'}
