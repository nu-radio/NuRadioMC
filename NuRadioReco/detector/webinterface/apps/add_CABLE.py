from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from plotly import subplots
import numpy as np
import plotly.graph_objs as go
import base64
from io import StringIO
import csv

from NuRadioReco.detector.detector_mongo import det
from NuRadioReco.detector.webinterface.utils.cable_helper import validate_Sdata, plot_Sparameters, sparameters_layout
from NuRadioReco.detector.webinterface.utils.table import get_table
from NuRadioReco.detector.webinterface.utils.units import str_to_unit
from NuRadioReco.detector.webinterface.app import app


table_name = "CABLE"

layout = html.Div([
    html.H3('Add S21 measurments for a CABLE', id='trigger'),
    html.Div(table_name, id='table-name'),
    dcc.Link('Go back to menu', href='/apps/menu'),
    html.Div([html.Div(dcc.Link('Add another CABLE measurement', href='/apps/add_CABLE', refresh=True), id=table_name + "-menu"),
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
    html.Div([html.Div("Select existing cable or enter new cable info:", style={'float':'left'}),
        # dcc.Dropdown(
        #     id='cable-list',
        #     options=[
        #         {'label': 'new CABLE', 'value': "new"}
        #     ],
        #     value="new",
        #     style={'width': '200px', 'float':'left'}
        # ),
        dcc.Dropdown(
            id='cable-color-input',
            options=[
                {'label': 'Orange (1m)', 'value': "1"},
                {'label': 'Blue (2m)', 'value': "2"},
                {'label': 'Green (3m)', 'value': "3"},
                {'label': 'White (4m)', 'value': "4"},
                {'label': 'Brown (5m)', 'value': "5"},
                {'label': 'Red/Grey (6m)', 'value': "6"}
            ],
            # value="new",
            # disabled=False,
            style={'width': '200px', 'float':'left'}
        ),
        dcc.Dropdown(
            id='cable-station-input',
            options=[
                {'label': 'Station 11 (Nanoq)', 'value': "11"},
                {'label': 'Station 12 (Terianniaq)', 'value': "12"},
                {'label': 'Station 13 (Ukaleq)', 'value': "13"},
                {'label': 'Station 14 (Tuttu)', 'value': "14"},
                {'label': 'Station 15 (Umimmak)', 'value': "15"},
                {'label': 'Station 21 (Amaroq)', 'value': "21"},
                {'label': 'Station 22 (Avinngaq)', 'value': "22"},
                {'label': 'Station 23 (Ukaliatsiaq)', 'value': "23"},
                {'label': 'Station 24 (Qappik)', 'value': "24"},
                {'label': 'Station 25 (Aataaq)', 'value': "25"}
            ],
            # value="new",
            # disabled=False,
            style={'width': '200px', 'float':'left'}
        ),
        dcc.Dropdown(
            id='cable-string-input',
            options=[
                {'label': 'A (Power String)', 'value': "A"},
                {'label': 'B (Helper String)', 'value': "B"},
                {'label': 'C (Helper String)', 'value': "C"}
            ],
            # value="new",
            # disabled=False,
            style={'width': '200px', 'float':'left'}
        ),
    ], style={'width':'100%', 'float': 'hidden'}),
    html.Br(),
    sparameters_layout,
    html.H4('', id=table_name + '-validation-global-output'),
    html.Div("false", id='validation-global', style={'display': 'none'}),
    html.Div([
        html.Button('insert to DB', id=table_name + '-button-insert', disabled=True),
    ], style={'width': "100%", "overflow": "hidden"}),
    html.Div(id='dd-output-container'),
    dcc.Graph(id='figure-cable', style={"height": "1000px", "width" : "100%"})
    ], id=table_name + "-main")])])

#
# @app.callback(
#     # [Output('cable-color-input', 'value'),  # 'disabled'),
#     #  Output('cable-station-input', 'value'),  # 'disabled'),
#     #  Output('cable-string-input', 'value'),  # 'disabled'),
#     [Output(table_name + "override-warning", "children")],
#     [Input('cable-list', 'value')])
# def enable_cable_name_input(value):
#     """
#     enable dropdowns for new cable name
#     """
#     if(value == "new"):
#         return False, ""
#     else:
#         return True, f"You are about to override the CABLE unit {value}!"
#
#
# @app.callback(
#     Output("cable-list", "options"),
#     [Input("trigger", "children")],
#     [State("cable-list", "options"),
#      State("table-name", "children")]
# )
# def update_dropdown_cable_names(n_intervals, options, table_name):
#     """
#     updates the dropdown menu with existing cable names from the database
#     """
#     if(get_table(table_name) is not None):
#         for cable_name in get_table(table_name).distinct("name"):
#             options.append(
#                 {"label": cable_name, "value": cable_name}
#             )
#         return options
#

@app.callback(
    [
        Output(table_name + "-validation-global-output", "children"),
        Output(table_name + "-validation-global-output", "style"),
        Output(table_name + "-validation-global-output", "data-validated"),
        Output(table_name + '-button-insert', 'disabled')
    ],
    [Input("validation-S21data-output", "data-validated"),
     Input('cable-color-input', 'value'),
     Input('cable-station-input', 'value'),
     Input('cable-string-input', 'value'),
     Input("function-test", "value")])
def validate_global(Sdata_validated, color, station, string, function_test):
    """
    validates all three inputs, this callback is triggered by the individual input validation
    """
    # new_cable_name = str(station) + str(string) + str(color)
    # if(cable_dropdown == "):
    #     return "cable name not set", {"color": "Red"}, False, True
    # if(cable_dropdown == "new" and (new_cable_name is None or new_cable_name == "")):
    #     return "cable name dropdown set to new but no new cable name was entered", {"color": "Red"}, False, True

    print(function_test)
    if('working' not in function_test):
        return "all inputs validated", {"color": "Green"}, True, False
    elif(Sdata_validated):
        return "all inputs validated", {"color": "Green"}, True, False

    return "input fields not validated", {"color": "Red"}, False, True


@app.callback([Output(table_name + '-main', 'style'),
               Output(table_name + '-menu', 'style')],
              [Input(table_name + '-button-insert', 'n_clicks')],
              [State('cable-color-input', 'value'),
               State('cable-station-input', 'value'),
               State('cable-string-input', 'value'),
             State('S21_mag_data', 'contents'),
             State('S21_phase_data', 'contents'),
             State('dropdown-frequencies', 'value'),
             State('dropdown-magnitude', 'value'),
             State('dropdown-phase', 'value'),
             State('separator', 'value'),
             State("function-test", "value"),
             State("protocol", "value")])
def insert_to_db(n_clicks, color, station, string, S21_mag_data, S21_phase_data,
                 unit_ff, unit_mag, unit_phase, sep, function_test, protocol):
    cable_name = str(station) + str(string) + str(color)
    print(f"n_clicks is {n_clicks}")
    if(not n_clicks is None):
        print("insert to db")
        # cable_name = cable_dropdown
        # if(cable_dropdown == "new"):
        #     cable_name = new_cable_name
        if('working' not in function_test):
            print(cable_name)
            det.CABLE_set_not_working(cable_name)

        else:
            mag_type, mag_string = S21_mag_data.split(',')
            Sm_datas = base64.b64decode(mag_string)
            Sm_data_io = StringIO(Sm_datas.decode('utf-8'))
            Sm_data = np.genfromtxt(Sm_data_io, skip_header=17, skip_footer=1, delimiter=sep).T
            phase_type, phase_string = S21_phase_data.split(',')
            Sp_datas = base64.b64decode(phase_string)
            Sp_data_io = StringIO(Sp_datas.decode('utf-8'))
            Sp_data = np.genfromtxt(Sp_data_io, skip_header=17, skip_footer=1, delimiter=sep).T
            Sm_data[0] *= str_to_unit[unit_ff]
            Sm_data[1] *= str_to_unit[unit_mag]
            Sp_data[1] *= str_to_unit[unit_phase]
            if('primary' not in function_test):
                primary_measurement = False
            else:
                primary_measurement = True
            print(cable_name, Sm_data, Sp_data[1])
            det.CABLE_add_Sparameters(cable_name, Sm_data, Sp_data, primary_measurement, protocol)

        return {'display': 'none'}, {}
    else:
        return {}, {'display': 'none'}
