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

from app import app

str_to_unit = {"GHz": units.GHz,
               "MHz": units.MHz,
               "Hz": units.Hz,
               "deg": units.deg,
               "rad": units.rad,
               "V": units.V,
               "mV": units.mV }

layout = html.Div([
    html.H3('Add new amplifier', id='trigger'),
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
            options=[{'label': x, 'value': x} for x in range(8)],
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
    Output("amp-board-list", "options"),
    [Input("trigger", "children")],
    [State("amp-board-list", "options")]
)
def update_dropdown_amp_names(n_intervals, options):
    """
    updates the dropdown menu with existing board names from the database
    """
    for amp_name in det.db.amp_boards.distinct("name"):
        options.append(
            {"label": amp_name, "value": amp_name}
        )
    return options


@app.callback(
    Output("override-warning", "children"),
    [Input("channel-id", "value")],
     [State("amp-board-list", "value"),
     State("S-dropdown", "value")])
def warn_override(channel_id, amp_name, S_parameter):
    """
    in case the user selects a channel that is already existing in the DB, a big warning is issued
    """
    existing_ids = det.db.amp_boards.distinct("channels.id", {"name": amp_name, "channels.S_parameter": S_parameter})
    if(channel_id in existing_ids):
        return f"You are about to override the {S_parameter} of channel {channel_id} of board {amp_name}!"
    else:
        return ""


@app.callback(
    [Output("validation-Sdata-output", "children"),
     Output("validation-Sdata-output", "style"),
    Output("validation-Sdata-output", "data-validated")],
    [Input('Sdata', 'value'),
     Input('dropdown-frequencies', 'value'),
     Input('dropdown-magnitude', 'value'),
     Input('dropdown-phase', 'value'),
     Input('separator', 'value')
     ])
def validate_Sdata(Sdata, unit_ff, unit_A, unit_phase, sep):
    """
    validates frequency array
    
    displays string with validation information for the user
    
    The outcome of the validataion (True/False) is saved in the 'data-validated' attribute to trigger other actions. 
    """
    if(Sdata != ""):
        try:
            S_data_io = StringIO(Sdata)
            S_data = np.genfromtxt(S_data_io, delimiter=sep).T
            S_data[0] *= str_to_unit[unit_ff]
            for i in range(4):
                S_data[1 + 2 * i] *= str_to_unit[unit_A]
                S_data[2 + 2 * i] *= str_to_unit[unit_phase]
            tmp = [f"you entered {len(S_data[0])} frequencies from {S_data[0].min()/units.MHz:.4g}MHz to {S_data[0].max()/units.MHz:.4g}MHz"]
            tmp.append(html.Br())
            S_names = ["S11", "S12", "S21", "S22"]
            for i in range(4):
                tmp.append(f"{S_names[i]} mag {len(S_data[1+i*2])} values within the range of {S_data[1+2*i].min()/units.V:.4g}V to {S_data[1+2*i].max()/units.V:.4g}V")
                tmp.append(html.Br())
                tmp.append(f"{S_names[i]} phase {len(S_data[2+i*2])} values within the range of {S_data[2+2*i].min()/units.degree:.1f}deg to {S_data[2+2*i].max()/units.degree:.1f}deg")
                tmp.append(html.Br())
            return tmp, {"color": "Green"}, True
        except:
    #         print(sys.exc_info())
            return f"{sys.exc_info()[0].__name__}:{sys.exc_info()[1]}", {"color": "Red"}, False
    else:
        return f"no data inserted", {"color": "Red"}, False


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


@app.callback(
    [Output("channel-id", "options"),
     Output("channel-id", "value")],
    [Input("trigger", "children"),
    Input("amp-board-list", "value"),
    Input("allow-override", "value")
    ]
)
def update_dropdown_channel_ids(n_intervals, amp_name, allow_override_checkbox):
    """
    disable all channels that are already in the database for that amp board and S parameter
    """
    print("update_dropdown_channel_ids")
    allow_override = False
    if 1 in allow_override_checkbox:
        allow_override = True

    existing_ids = det.db.amp_boards.distinct("channels.id", {"name": amp_name, "channels.S_parameter": {"$in": ["S11", "S12", "S21", "S22"]}})
    print(f"existing ids for amp {amp_name}: {existing_ids}")
    options = []
    for i in range(24):
        if(i in existing_ids):
            if(allow_override):
                options.append({"label": f"{i} (already exists)", "value": i})
            else:
                options.append({"label": i, "value": i, 'disabled': True})
        else:
            options.append({"label": i, "value": i})
    return options, ""


@app.callback(
    Output('new-board-input', 'disabled'),
    [Input('amp-board-list', 'value')])
def enable_board_name_input(value):
    if(value == "new"):
        return False
    else:
        return True


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


@app.callback(
    Output('figure-amp', 'figure'),
    [Input("validation-Sdata-output", "data-validated")],
    [State('Sdata', 'value'),
             State('dropdown-frequencies', 'value'),
             State('dropdown-magnitude', 'value'),
             State('dropdown-phase', 'value'),
             State('separator', 'value')])
def display_value(val_Sdata, Sdata, unit_ff, unit_mag, unit_phase, sep):
    print("display_value")
    if(val_Sdata):
        S_data_io = StringIO(Sdata)
        S_data = np.genfromtxt(S_data_io, delimiter=sep).T
        S_data[0] *= str_to_unit[unit_ff]
        for i in range(4):
            S_data[1 + 2 * i] *= str_to_unit[unit_mag]
            S_data[2 + 2 * i] *= str_to_unit[unit_phase]
        fig = subplots.make_subplots(rows=4, cols=2)
        for i in range(4):
            fig.append_trace(go.Scatter(
                        x=S_data[0] / units.MHz,
                        y=S_data[i * 2 + 1] / units.V,
                        opacity=0.7,
                        marker={
                            'color': "blue",
                            'line': {'color': "blue"}
                        },
                        name='magnitude'
                    ), i + 1, 1)
            fig.append_trace(go.Scatter(
                        x=S_data[0] / units.MHz,
                        y=S_data[i * 2 + 2] / units.deg,
                        opacity=0.7,
                        marker={
                            'color': "blue",
                            'line': {'color': "blue"}
                        },
                        name='phase'
                    ), i + 1, 2)
        fig['layout']['xaxis1'].update(title='frequency [MHz]')
        fig['layout']['yaxis1'].update(title='magnitude [V]')
        fig['layout']['yaxis2'].update(title='phase [deg]')
        fig['layout']['xaxis2'].update(title='frequency [MHz]')
        return fig
    else:
        return {"data": []}
