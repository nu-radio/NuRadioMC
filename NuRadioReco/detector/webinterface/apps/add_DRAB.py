import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from NuRadioReco.utilities import units
from plotly import subplots
import numpy as np
import plotly.graph_objs as go
import json
import base64
import sys
from io import StringIO
import csv

from NuRadioReco.detector import detector_mongo as det
from NuRadioReco.detector.webinterface.utils.sparameter_helper import validate_Sdata, update_dropdown_amp_names, enable_board_name_input, plot_Sparameters, sparameters_layout
from NuRadioReco.detector.webinterface.utils.table import get_table
from NuRadioReco.detector.webinterface.utils.units import str_to_unit
from NuRadioReco.detector.webinterface.app import app

table_name = "DRAB"
number_of_channels = 4

layout = html.Div([
    html.H3('Add S parameter measurement of DRAB unit', id='trigger'),
    html.Div(table_name, id='table-name'),
    dcc.Link('Go back to menu', href='/apps/menu'),
    html.Div([html.Div(dcc.Link('Add another DRAB unit measurement', href='/apps/add_DRAB'), id=table_name + "-menu"),
              html.Div([
    html.H3('', id=table_name + 'override-warning', style={"color": "Red"}),
    html.Div([
    dcc.Checklist(
        id="allow-override",
        options=[
            {'label': 'Allow override of existing entries', 'value': 1}
        ],
        value=[])
    ], style={'width':'100%', 'float': 'hidden'}),
    html.Br(),
    html.Br(),
    html.Div([html.Div("Select existing board or enter unique name of new board:", style={'float':'left'}),
        dcc.Dropdown(
            id='drab-list',
            options=[
                {'label': 'new DRAB', 'value': "new"}
            ],
            value="new",
            style={'width': '200px', 'float':'left'}
        ),
        dcc.Input(id="new-drab-input",
                  disabled=False,
                  placeholder='new unique DRAB name',
                  style={'width': '200px',
                         'float': 'left'}),
        dcc.Dropdown(
            id=table_name + 'channel-id',
            options=[{'label': x, 'value': x} for x in range(number_of_channels)],
            placeholder='channel-id',
            style={'width': '200px', 'float':'left'}
        ),
        dcc.Dropdown(
            id='IGLU-id',
            options=[],
            value='Golden_IGLU',
            style={'width': '200px', 'float':'left'}
        ),
        dcc.Dropdown(
            id='temperature-list',
            options=[
                {'label': 'room temp (20* C)', 'value': 20},
                {'label': '-50*C', 'value': -50},
                {'label': '-40*C', 'value': -40},
                {'label': '-30*C', 'value': -30},
                {'label': '-20*C', 'value': -20},
                {'label': '-10*C', 'value': -10},
                {'label': '0*C', 'value': 0},
                {'label': '10*C', 'value': 10},
                {'label': '30*C', 'value': 20},
                {'label': '40*C', 'value': 30},
            ],
            value=20,
            style={'width': '200px', 'float':'left'})
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
    dcc.Graph(id='figure-amp', style={"height": "1000px", "width" : "100%"})
    ], id=table_name + "-main")])])


@app.callback(
    [Output('new-drab-input', 'disabled'),
     Output(table_name + "override-warning", "children")],
    [Input('drab-list', 'value')])

def enable_board_name_input(value):
    """
    enable text field for new DRAB unit
    """
    if(value == "new"):
        return False, ""
    else:
        return True, f"You are about to override the DRAB unit {value}!"


@app.callback(
    Output("drab-list", "options"),
    [Input("trigger", "children")],
    [State("drab-list", "options"),
     State("table-name", "children")]
)
def update_dropdown_drab_names(n_intervals, options, table_name):
    """
    updates the dropdown menu with existing board names from the database
    """
    if(get_table(table_name) is not None):
        for amp_name in get_table(table_name).distinct("name"):
            options.append(
                {"label": amp_name, "value": amp_name}
            )
        return options

@app.callback(
    Output("IGLU-id", "options"),
    [Input("trigger", "children")],
    [State("table-name", "children")]
)
def update_dropdown_iglu_names(n_intervals, table_name):
    """
    updates the dropdown menu with existing board names from the database
    """
    if(get_table(table_name) is not None):
        options = []
        for amp_name in get_table("IGLU").distinct("name"):
            options.append(
                {"label": amp_name, "value": amp_name}
            )
        options.append({"label": "without IGLU", "value": "wo_Iglu"})
        return options


@app.callback(
    [
        Output(table_name + "-validation-global-output", "children"),
        Output(table_name + "-validation-global-output", "style"),
        Output(table_name + "-validation-global-output", "data-validated"),
        Output(table_name + '-button-insert', 'disabled')
    ],
    [Input("validation-Sdata-output", "data-validated"),
     Input('drab-list', 'value'),
     Input('new-drab-input', 'value'),
     Input("function-test", "value")])
def validate_global(Sdata_validated, board_dropdown, new_board_name, function_test):
    """
    validates all three inputs, this callback is triggered by the individual input validation
    """
    if(board_dropdown == ""):
        return "board name not set", {"color": "Red"}, False, True
    if(board_dropdown == "new" and (new_board_name is None or new_board_name == "")):
        return "board name dropdown set to new but no new board name was entered", {"color": "Red"}, False, True

    print(function_test)
    if('working' not in function_test):
        return "all inputs validated", {"color": "Green"}, True, False
    elif(Sdata_validated):
        return "all inputs validated", {"color": "Green"}, True, False

    return "input fields not validated", {"color": "Red"}, False, True


@app.callback([Output(table_name + '-main', 'style'),
               Output(table_name + '-menu', 'style')],
              [Input(table_name + '-button-insert', 'n_clicks')],
              [State('drab-list', 'value'),
               State('new-drab-input', 'value'),
             State('Sdata', 'contents'),
             State('dropdown-frequencies', 'value'),
             State('dropdown-magnitude', 'value'),
             State('dropdown-phase', 'value'),
             State(table_name + "channel-id", "value"),
             State("IGLU-id", "value"),
             State('separator', 'value'),
             State('temperature-list', 'value'),
             State("function-test", "value")])
def insert_to_db(n_clicks, board_dropdown, new_board_name, contents, unit_ff, unit_mag, unit_phase, channel_id, iglu_id, sep, temp, function_test):
    print(f"n_clicks is {n_clicks}")
    if(not n_clicks is None):
        print("insert to db")
        board_name = board_dropdown
        if(board_dropdown == "new"):
            board_name = new_board_name
        if('working' not in function_test):
            det.DRAB_set_not_working(board_name)
        else:
            content_type, content_string = contents.split(',')
            S_datas = base64.b64decode(content_string)
            S_data_io = StringIO(S_datas.decode('utf-8'))
            S_data = np.genfromtxt(S_data_io, skip_header=7, skip_footer=1, delimiter=sep).T
            S_data[0] *= str_to_unit[unit_ff]
            for i in range(4):
                S_data[1 + 2 * i] *= str_to_unit[unit_mag]
                S_data[2 + 2 * i] *= str_to_unit[unit_phase]
            print("channelid" + str(channel_id))
            print(board_name, S_data)
            det.DRAB_add_Sparameters(board_name, channel_id, iglu_id, temp, S_data)

        return {'display': 'none'}, {}
    else:
        return {}, {'display': 'none'}
