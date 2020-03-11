import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from NuRadioReco.utilities import units
from plotly import subplots
import numpy as np
import plotly.graph_objs as go
import json
import sys
from io import StringIO
import csv

from NuRadioReco.detector import detector_mongo as det
from NuRadioReco.detector.webinterface.apps.sparameter_helper import validate_Sdata, warn_override, update_dropdown_amp_names, update_dropdown_channel_ids, enable_board_name_input, plot_Sparameters

from app import app

str_to_unit = {"GHz": units.GHz,
               "MHz": units.MHz,
               "Hz": units.Hz,
               "deg": units.deg,
               "rad": units.rad,
               "V": units.V,
               "mV": units.mV }

number_of_channels = 5  # define number of channels for surface board

layout = html.Div([
    html.H3('Add new amplifier', id='trigger'),
    html.Div("amp_boards", id='table-name'),
    html.Div(number_of_channels, id='number-of-channels'),
    dcc.Link('Go back to menu', href='/apps/menu'),
    html.H3('', id='override-warning', style={"color": "Red"}),
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
                         'float': 'left'}),
        dcc.Dropdown(
            id='channel-id',
            options=[{'label': x, 'value': x} for x in range(number_of_channels)],
            placeholder='channel-id',
            style={'width': '200px', 'float':'left'}
        ),
    ], style={'width':'100%', 'float': 'hidden'}),
    html.Br(),
    html.Br(),
    html.Div("specify data format:"),
    dcc.Dropdown(
            id='separator',
            options=[
                {'label': 'comma separated ","', 'value': ','},
                     ],
            clearable=False,
            value=",",
            style={'width': '200px', 'float':'left'}
        ),
    html.Div("units"),
    dcc.Dropdown(
        id='dropdown-frequencies',
        options=[
            {'label': 'GHz', 'value': "GHz"},
            {'label': 'MHz', 'value': "MHz"},
            {'label': 'Hz', 'value': "Hz"}
        ],
        value="Hz",
        style={'width': '20%',
#                'float': 'left'
        }
    ),
    dcc.Dropdown(
            id='dropdown-magnitude',
            options=[
                {'label': 'V', 'value': "V"},
                {'label': 'mV', 'value': "mV"}
            ],
            value="V",
            style={'width': '20%',
#                    'float': 'left'}
                   }
        ),
    dcc.Dropdown(
            id='dropdown-phase',
            options=[
                {'label': 'degree', 'value': "deg"},
                {'label': 'rad', 'value': "rad"}
            ],
            value="deg",
            style={'width': '20%',
#                    'float': 'left'}
            }
        ),
    html.Br(),
    html.Br(),
    html.Div([
        dcc.Textarea(
            id='Sdata',
            placeholder='data array from spectrum analyzser of the form Freq, S11(MAG)    S11(DEG)    S12(MAG)    S12(DEG)    S21(MAG)    S21(DEG)    S22(MAG)    S22(DEG)',
            value='',
            style={'width': '60%',
                   'float': 'left'}
        ),
    ], style={'width':'100%', 'float': 'hidden'}),
    html.Br(),
    html.Div('', id='validation-Sdata-output', style={'whiteSpace': 'pre-wrap'}),
    html.H4('', id='validation-global-output'),
    html.Div("false", id='validation-global', style={'display': 'none'}),
    html.Div([
        html.Button('insert to DB', id='button-insert', disabled=True),
    ], style={'width':"100%", "overflow": "hidden"}),
    html.Div(id='dd-output-container'),
    dcc.Graph(id='figure-amp', style={"height": "1000px", "width" : "100%"})
])


@app.callback(
    [
        Output("validation-global-output", "children"),
        Output("validation-global-output", "style"),
        Output("validation-global-output", "data-validated"),
        Output('button-insert', 'disabled')
    ],
    [Input("validation-Sdata-output", "data-validated"),
     Input('amp-board-list', 'value'),
     Input('new-board-input', 'value'),
     Input("channel-id", "value")])
def validate_global(Sdata_validated, board_dropdown, new_board_name, channel_id
                    ):
    """
    validates all three inputs, this callback is triggered by the individual input validation
    """
    if(board_dropdown == ""):
        return "board name not set", {"color": "Red"}, False, True
    if(board_dropdown == "new" and (new_board_name is None or new_board_name == "")):
        return "board name dropdown set to new but no new board name was entered", {"color": "Red"}, False, True
    if(channel_id not in range(100)):
        return "no channel id selected", {"color": "Red"}, False, True

    if(Sdata_validated):
        return "all inputs validated", {"color": "Green"}, True, False

    return "input fields not validated", {"color": "Red"}, False, True


@app.callback(Output('url', 'pathname'),
              [Input('button-insert', 'n_clicks_timestamp')],
              [State('amp-board-list', 'value'),
               State('new-board-input', 'value'),
             State('Sdata', 'value'),
             State('dropdown-frequencies', 'value'),
             State('dropdown-magnitude', 'value'),
             State('dropdown-phase', 'value'),
             State("channel-id", "value"),
             State('separator', 'value')])
def insert_to_db(n_clicks, board_dropdown, new_board_name, Sdata, unit_ff, unit_mag, unit_phase, channel_id, sep):
    print("insert to db")
    S_data_io = StringIO(Sdata)
    S_data = np.genfromtxt(S_data_io, delimiter=sep).T
    S_data[0] *= str_to_unit[unit_ff]
    for i in range(4):
        S_data[1 + 2 * i] *= str_to_unit[unit_mag]
        S_data[2 + 2 * i] *= str_to_unit[unit_phase]

    board_name = board_dropdown
    if(board_dropdown == "new"):
        board_name = new_board_name
    print(board_name, channel_id, S_data)
    det.insert_amp_board_channel_Sparameters(board_name, channel_id, S_data)

    return "/apps/menu"

