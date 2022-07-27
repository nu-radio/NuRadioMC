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

from NuRadioReco.detector.detector_mongo import det
# from NuRadioReco.detector.detector_mongo import Detector
# from NuRadioReco.detector.webinterface import config
# from NuRadioReco.detector.webinterface.utils.sparameter_helper import validate_Sdata, update_dropdown_amp_names, enable_board_name_input, plot_Sparameters, sparameters_layout
from NuRadioReco.detector.webinterface.utils.Hpol_helper import validate_Sdata, plot_Sparameters, sparameters_layout
from NuRadioReco.detector.webinterface.utils.table import get_table
from NuRadioReco.detector.webinterface.utils.units import str_to_unit
from NuRadioReco.detector.webinterface.app import app


table_name = "HPol"

layout = html.Div([
    html.H3('Add a measurment of an HPol', id='trigger'),
    html.Div(table_name, id='table-name'),
    dcc.Link('Go back to menu', href='/apps/menu'),
    html.Div([html.Div(dcc.Link('Add another HPol measurement', href='/apps/add_HPol', refresh=True), id=table_name + "-menu"),
              html.Div([
    html.H3('', id=table_name + 'override-warning', style={'color': 'Red'}),
    html.Div([
        dcc.Checklist(
            id='allow-override',
            options=[{'label': 'Allow override existing entries', 'value': 1}],
            value=[])
    ], style={'width':'20%', 'float':'hidden'}),
    html.Br(),
    html.Br(),
    html.Div([html.Div('Select existing antenna or enter unique name of new antenna:', style={'float':'left'}),
        dcc.Dropdown(
            id='HPol-list',
            options=[{'label':'new HPol', 'value':'new'}],
            value='new',
            style={'width':'200px', 'float':'left'}),
        dcc.Input(
            id='new-HPol-input',
            disabled=False,
            placeholder='new unique Hpol name',
            style={'width': '200px', 'float':'left'}),
    ], style={'width':'100%', 'float':'hidden'}),
    html.Br(),
    html.Br(),
    sparameters_layout,
    html.H4('', id=table_name + '-validation-global-output'),
    html.Div([html.Button('insert to DB', id=table_name+'-button-insert', disabled=True)],
             style={'width': '100%', 'overflow': 'hidden'}
             ),
    html.Div(id='dd-output-container'),
    dcc.Graph(id='figure-HPol', style={'height': '1000px', 'width': '100%'})
    ], id=table_name + "-main")]),
    ])

@app.callback(
    [
        Output('new-HPol-input', 'disabled'),
        Output(table_name + 'override-warning', 'children')
    ],
    Input('HPol-list', 'value')
)
def enable_HPol_name_input(value):
    """
    enable text field to enter a new Hpol unit name
    """
    if value == 'new':
        # enables the text field to enter a new HPol name
        return False, ''
    else:
        # disables the text field and give a warning that you will override HPols units
        return True, f'You are about to override the HPol unit {value}!'

@app.callback(
    Output('HPol-list', 'options'),
    Input('trigger', 'children'),
    [
        State('HPol-list', 'options'),
        State('table-name', 'children'),
    ]
)
def update_dropdown_HPol_names(n_intervals, options, current_table_name):
    """
    updates the Dropdown menu with the existing antenna names from the database
    """
    if get_table(current_table_name) is not None:
        # get_table.distinct('name') gives back a list and all elements of that list are added to options (the dropdown menu)
        for HPol_name in get_table(current_table_name).distinct('name'):
            options.append({'label': HPol_name, 'value': HPol_name})
        return options


@app.callback(
    [
        Output(table_name + '-validation-global-output', 'children'),
        Output(table_name + '-validation-global-output', 'style'),
        Output(table_name + '-validation-global-output', 'data-validated'),
        Output(table_name + '-button-insert', 'disabled'),
    ],
    [
        Input(table_name + "-validation-Sdata-output", 'data-validated'),
        Input('HPol-list', 'value'),
        Input('new-HPol-input', 'value'),
        Input('function-test', 'value'),
    ]
)
def validate_global(Sdata_validated, HPol_dropdown, new_HPol_name, function_test):
    """
    validates all three inputs, this callback is triggered by the individual input validation
    """
    if HPol_dropdown == "":
        return 'antenna name not set', {'color':'Red'}, False, True
    if HPol_dropdown == 'new' and (new_HPol_name is None or new_HPol_name == ''):
        return 'antenna name dropdown set to new but no new antenna name was entered', {'color':'Red'}, False, True

    print(function_test)
    if 'working' not in function_test:
        return 'all inputs validated', {'color': 'Green'}, True, False
    elif Sdata_validated:
        return 'all inputs validated', {'color': 'Green'}, True, False

    return 'input fields not validated', {'color': 'Red'}, False, True


@app.callback(
    [
        Output(table_name + '-main', 'style'),
        Output(table_name + '-menu', 'style')
    ],
    [
        Input(table_name + '-button-insert', 'n_clicks')
    ],
    [
        State('HPol-list', 'value'),
        State('new-HPol-input', 'value'),
        State('Sdata', 'contents'),
        State('dropdown-frequencies', 'value'),
        State('dropdown-magnitude', 'value'),
        State('separator', 'value'),
        State('function-test', 'value'),
        State('protocol', 'value')
    ]
)
def insert_to_db(n_clicks, HPol_dropdown, new_HPol_name, contents, unit_ff, unit_mag, sep, function_test, protocol):
    print(f'n_clicks is {n_clicks}')
    if not n_clicks is None:
        print('insert to db')
        HPol_name = HPol_dropdown
        if HPol_dropdown == 'new':
            HPol_name = new_HPol_name
        if 'working' not in function_test:
            print(HPol_name)
            det.HPol_set_not_working(HPol_name)
        else:
            content_type, content_string = contents.split(',')
            S_datas = base64.b64decode(content_string)
            S_data_io = StringIO(S_datas.decode('utf-8'))
            S_data = np.genfromtxt(S_data_io, skip_header=30, skip_footer=1, delimiter=sep).T
            S_data[0] *= str_to_unit[unit_ff]
            S_data[1] *= str_to_unit[unit_mag]
            if 'primary' not in function_test:
                primary_measurement = False
            else:
                primary_measurement = True
            print(HPol_name, S_data)
            det.HPol_add_Sparameters(HPol_name, S_data, primary_measurement, protocol, unit_ff, unit_mag)

        return {'display': 'none'}, {}
    else:
        return {}, {'display': 'none'}