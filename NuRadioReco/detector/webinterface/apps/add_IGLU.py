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
from datetime import datetime

from NuRadioReco.detector.detector_mongo import det
from NuRadioReco.detector.webinterface.utils.sparameter_helper import validate_Sdata, update_dropdown_amp_names, enable_board_name_input, plot_Sparameters, sparameters_layout
from NuRadioReco.detector.webinterface.utils.table import get_table
from NuRadioReco.detector.webinterface.utils.units import str_to_unit
from NuRadioReco.detector.webinterface.app import app


number_of_channels = 1  # define number of channels for IGLU
table_name = "IGLU"

layout = html.Div([
    html.H3('Add S parameter measurement of IGLU board', id='trigger'),
    html.Div(table_name, id='table-name'),
    html.Div(number_of_channels, id='number-of-channels'),
    dcc.Link('Go back to menu', href='/apps/menu'),
    html.Div([html.Div(dcc.Link('Add another IGLU board measurement', href='/apps/add_IGLU', refresh=True), id=table_name + "-menu"),
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
            id='amp-board-list',
            options=[
                {'label': 'new board', 'value': "new"}
            ],
            value="",
            style={'width': '200px', 'float':'left'}
        ),
        dcc.Input(id="new-board-input",
                  disabled=True,
                  placeholder='new unique board name',
                  style={'width': '200px',
                         'float': 'left'}
        ),

        dcc.Dropdown(
            id='DRAB-id',
            options=[],
            value='Golden_DRAB',
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
                {'label': '10*C', 'value': 0},
                {'label': '30*C', 'value': 0},
                {'label': '40*C', 'value': 0},
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
    Output(table_name + "override-warning", "children"),
    [Input("DRAB-id", "value")],
     [State("amp-board-list", "value"),
      State("table-name", "children")])
def warn_override(drab_id, amp_name, table_name):
    """
    in case the user selects a channel that is already existing in the DB, a big warning is issued
    """
    if(drab_id is None):
        return ""
    if(drab_id == "wo_DRAB"):
        existing_ids = get_table(table_name).distinct("channels.id", {"name": amp_name, "channels.S_parameter_wo_DRAB": {"$in": ["S11", "S12", "S21", "S22"]}})
    else:
        existing_ids = get_table(table_name).distinct("channels.id", {"name": amp_name, "channels.S_parameter_DRAB": {"$in": ["S11", "S12", "S21", "S22"]}})


# @app.callback(
#     [Input("trigger", "children"),
#     Input("amp-board-list", "value"),
#     Input("allow-override", "value"),
#     Input("DRAB-id", "value")
#     ],
#     [State("table-name", "children"),
#      State("number-of-channels", "children")]
#)
# def update_dropdown_channel_ids(n_intervals, amp_name, allow_override_checkbox, drab_id, table_name, number_of_channels):
#     """
#     disable all channels that are already in the database for that amp board and S parameter
#     """
#     number_of_channels = int(number_of_channels)
#     print("update_dropdown_channel_ids")
#     allow_override = False
#     if 1 in allow_override_checkbox:
#         allow_override = True
#
#     if(drab_id is None):
#         return [], ""
#     if(drab_id == "wo_DRAB"):
#         existing_ids = get_table(table_name).distinct("channels.id", {"name": amp_name, "channels.S_parameter_wo_DRAB": {"$in": ["S11", "S12", "S21", "S22"]}})
#     else:
#         existing_ids = get_table(table_name).distinct("channels.id", {"name": amp_name, "channels.S_parameter_DRAB": {"$in": ["S11", "S12", "S21", "S22"]}})
#     print(f"existing ids for amp {amp_name}: {existing_ids}")
#     options = []
#     for i in range(number_of_channels):
#         if(i in existing_ids):
#             if(allow_override):
#                 options.append({"label": f"{i} (already exists)", "value": i})
#             else:
#                 options.append({"label": i, "value": i, 'disabled': True})
#         else:
#             options.append({"label": i, "value": i})
#     return options, ""


@app.callback(
    Output("DRAB-id", "options"),
    [Input("trigger", "children")],
    [State("table-name", "children")]
)
def update_dropdown_drab_names(n_intervals, table_name):
    """
    updates the dropdown menu with existing board names from the database
    """
    if(get_table(table_name) is not None):
        options = []
        for amp_name in get_table("DRAB").distinct("name"):
            options.append(
                {"label": amp_name, "value": amp_name}
            )
        options.append({"label": "without DRAB", "value": "wo_DRAB"})
        return options


@app.callback(
    [
        Output(table_name + "-validation-global-output", "children"),
        Output(table_name + "-validation-global-output", "style"),
        Output(table_name + "-validation-global-output", "data-validated"),
        Output(table_name + '-button-insert', 'disabled')
    ],
    [Input("validation-Sdata-output", "data-validated"),
     Input('amp-board-list', 'value'),
     Input('new-board-input', 'value'),
     Input("DRAB-id", "value"),
     Input("function-test", "value")])
def validate_global(Sdata_validated, board_dropdown, new_board_name, drab_id, function_test):
    """
    validates all three inputs, this callback is triggered by the individual input validation
    """
    print(f"validate global drabid = {drab_id}")
    if(board_dropdown == ""):
        return "board name not set", {"color": "Red"}, False, True
    if(board_dropdown == "new" and (new_board_name is None or new_board_name == "")):
        return "board name dropdown set to new but no new board name was entered", {"color": "Red"}, False, True
    if(drab_id is None):
        return "no DRAB unit selected", {"color": "Red"}, False, True

    print(function_test)
    if('working' not in function_test):
        return "all inputs validated", {"color": "Green"}, True, False
    elif(Sdata_validated):
        return "all inputs validated", {"color": "Green"}, True, False

    return "input fields not validated", {"color": "Red"}, False, True


@app.callback([Output(table_name + '-main', 'style'),
               Output(table_name + '-menu', 'style')],
              [Input(table_name + '-button-insert', 'n_clicks')],
              [State('amp-board-list', 'value'),
               State('new-board-input', 'value'),
             State('Sdata', 'contents'),
             State('dropdown-frequencies', 'value'),
             State('dropdown-magnitude', 'value'),
             State('dropdown-phase', 'value'),
             State("DRAB-id", "value"),
             State('separator', 'value'),
             State('temperature-list', 'value'),
             State("function-test", "value"),
             State("group_delay_corr", "value"),
             State("protocol", "value")])
def insert_to_db(n_clicks, board_dropdown, new_board_name, contents, unit_ff,
                 unit_mag, unit_phase, drab_id, sep, temp, function_test, corr_group_delay, protocol):
    print(f"n_clicks is {n_clicks}")
    if(not n_clicks is None):
        print("insert to db")
        board_name = board_dropdown
        if(board_dropdown == "new"):
            board_name = new_board_name
        if('working' not in function_test):
            det.IGLU_board_channel_set_not_working(board_name)
        else:
            content_type, content_string = contents.split(',')
            S_datas = base64.b64decode(content_string)
            S_data_io = StringIO(S_datas.decode('utf-8'))
            header = []
            for i in range(7):
                header.append(S_data_io.readline())
            date_string = header[2]
            date_string_cropped = date_string[7:-2]
            date_string_fixed = date_string_cropped.replace(",", "")
            measurement_time = datetime.strptime(date_string_fixed, '%A %B %d %Y %X')
            if('primary' not in function_test):
                primary_measurement = False
            else:
                primary_measurement = True
                det.IGLU_remove_primary(board_name)
            S_data = np.genfromtxt(S_data_io, skip_footer=1, delimiter=sep).T
            S_data[0] *= str_to_unit[unit_ff]
            for i in range(4):
                S_data[1 + 2 * i] *= str_to_unit[unit_mag]
                S_data[2 + 2 * i] *= str_to_unit[unit_phase]
            if corr_group_delay is None:
                correction = 0
            else:
                correction = corr_group_delay
            time_delay = [0, 0, correction * units.ns, 0]
            if(drab_id == "wo_DRAB"):
                det.IGLU_board_channel_add_Sparameters_without_DRAB(board_name,
                temp, S_data, measurement_time, primary_measurement, time_delay, protocol)
            else:
                det.IGLU_board_channel_add_Sparameters_with_DRAB(board_name,
                drab_id, temp, S_data, measurement_time, primary_measurement, time_delay, protocol)
        return {'display': 'none'}, {}
    else:
        return {}, {'display': 'none'}
