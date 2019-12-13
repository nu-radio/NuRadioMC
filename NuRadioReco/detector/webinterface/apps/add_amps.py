import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from NuRadioReco.utilities import units
from plotly import subplots
import numpy as np
import plotly.graph_objs as go
import json
import sys

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
    html.Div([html.Div("What type of measurement are you adding?:", style={'float':'left'}),
    dcc.Dropdown(
        id='S-dropdown',
        options=[
            {'label': 'S11', 'value': "S11"},
            {'label': 'S12', 'value': "S12"},
            {'label': 'S21', 'value': "S21"},
            {'label': 'S22', 'value': "S22"}
        ],
        placeholder="select S parameter",
        value="S12",
        style={'width': '200px', 'float':'left'}
    ),
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
                {'label': 'new value per line "\\n"', 'value': '\n'}
                     ],
            clearable=False,
            value=",",
            style={'width': '200px', 'float':'left'}
        ),
    html.Br(),
    html.Br(),
    html.Div([
        dcc.Textarea(
            id='frequencies',
            placeholder='frequencies',
            value='',
            style={'width': '60%',
                   'float': 'left'}
        ),
        dcc.Dropdown(
            id='dropdown-frequencies',
            options=[
                {'label': 'GHz', 'value': "GHz"},
                {'label': 'MHz', 'value': "MHz"},
                {'label': 'Hz', 'value': "Hz"}
            ],
            value="GHz",
            style={'width': '40%',
                   'float': 'left'}
        )
    ], style={'width':'100%', 'float': 'hidden'}),
    html.Div(id='validation-frequencies-output'),
    html.Div([
        dcc.Textarea(
            id='magnitude',
            placeholder='magnitudes',
            value='',
            style={'width': '60%',
                   'float': 'left'}
        ),
        dcc.Dropdown(
            id='dropdown-magnitude',
            options=[
                {'label': 'V', 'value': "V"},
                {'label': 'mV', 'value': "mV"}
            ],
            value="V",
            style={'width': '40%',
                   'float': 'left'}
        )
    ], style={'width':'100%'}),
    html.Div(id='validation-magnitude-output'),
    html.Div([
        dcc.Textarea(
            id='phases',
            placeholder='phase',
            value='',
            style={'width': '60%',
                   'float': 'left'}
        ),
        dcc.Dropdown(
            id='dropdown-phase',
            options=[
                {'label': 'degree', 'value': "deg"},
                {'label': 'rad', 'value': "rad"}
            ],
            value="deg",
            style={'width': '40%',
                   'float': 'left'}
        )
    ], style={'width':'100%'}),
    html.Div(id='validation-phase-output'),
    html.Br(),
    html.H4('', id='validation-global-output'),
    html.Div("false", id='validation-global', style={'display': 'none'}),
    html.Div([
        html.Button('insert to DB', id='button-insert', disabled=True),
    ], style={'width':"100%", "overflow": "hidden"}),
    html.Div(id='dd-output-container'),
    dcc.Graph(id='figure-amp')
])


@app.callback(
    Output("amp-board-list", "options"),
    [Input("trigger", "value")],
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
    [Output("validation-frequencies-output", "children"),
     Output("validation-frequencies-output", "style"),
    Output("validation-frequencies-output", "data-validated")],
    [Input('frequencies', 'value'),
     Input('dropdown-frequencies', 'value'),
     Input('separator', 'value')
     ])
def validate_frequencies(ff, unit_ff, sep):
    """
    validates frequency array
    
    displays string with validation information for the user
    
    The outcome of the validataion (True/False) is saved in the 'data-validated' attribute to trigger other actions. 
    """
    try:
        ff = np.array([float(i) for i in ff.split(sep)]) * str_to_unit[unit_ff]

        return f"you entered {len(ff)} frequencies from {ff.min()/units.MHz:.4g}MHz to {ff.max()/units.MHz:.4g}MHz", {"color": "Green"}, True
    except:
        return f"{sys.exc_info()[0].__name__}:{sys.exc_info()[1]}", {"color": "Red"}, False


@app.callback(
    [Output("validation-magnitude-output", "children"),
     Output("validation-magnitude-output", "style"),
    Output("validation-magnitude-output", "data-validated")],
    [Input('magnitude', 'value'),
     Input('dropdown-magnitude', 'value'),
     Input('separator', 'value')])
def validate_mag(mag, unit_mag, sep):
    """
    validates magnitude array
    
    displays string with validation information for the user
    
    The outcome of the validataion (True/False) is saved in the 'data-validated' attribute to trigger other actions. 
    """
    try:
        mag = np.array([float(i) for i in mag.split(sep)]) * str_to_unit[unit_mag]
        return f"you entered {len(mag)} values within the range of {mag.min()/units.V:.4g}V to {mag.max()/units.V:.4g}V", {"color": "Green"}, True
    except:
        return f"{sys.exc_info()[0].__name__}:{sys.exc_info()[1]}", {"color": "Red"}, False


@app.callback(
    [Output("validation-phase-output", "children"),
     Output("validation-phase-output", "style"),
     Output("validation-phase-output", "data-validated")],
    [Input('phases', 'value'),
     Input('dropdown-phase', 'value'),
     Input('separator', 'value')])
def validate_phase(phase, unit_phase, sep):
    """
    validates phase array
    
    displays string with validation information for the user
    
    The outcome of the validataion (True/False) is saved in the 'data-validated' attribute to trigger other actions. 
    """
    try:
        phase = np.array([float(i) for i in phase.split(sep)]) * str_to_unit[unit_phase]
        return f"you entered {len(phase)} values within the range of {phase.min()/units.degree:.1f}degree to {phase.max()/units.degree:.1f}degree", {"color": "Green"}, True
    except:
        return f"{sys.exc_info()[0].__name__}:{sys.exc_info()[1]}", {"color": "Red"}, False


@app.callback(
    [
        Output("validation-global-output", "children"),
        Output("validation-global-output", "style"),
        Output("validation-global-output", "data-validated"),
        Output('button-insert', 'disabled')
    ],
    [Input("validation-frequencies-output", "data-validated"),
     Input("validation-magnitude-output", "data-validated"),
     Input("validation-phase-output", "data-validated"),
     Input('amp-board-list', 'value'),
     Input('new-board-input', 'value'),
     Input("channel-id", "value")],
    [State('frequencies', 'value'),
    State('magnitude', 'value'),
    State('phases', 'value'),
    State('dropdown-frequencies', 'value'),
    State('dropdown-magnitude', 'value'),
    State('dropdown-phase', 'value'),
    State('separator', 'value')])
def validate_global(ff_validated, mag_validated, phase_validated, board_dropdown, new_board_name, channel_id,
                    ff, mag, phase, unit_ff, unit_mag, unit_phase, sep
                    ):
    """
    validates all three inputs, this callback is triggered by the individual input validation
    """
    print("validate_global")
    if(board_dropdown == ""):
        return "board name not set", {"color": "Red"}, False, True
    if(board_dropdown == "new" and (new_board_name is None or new_board_name == "")):
        return "board name dropdown set to new but no new board name was entered", {"color": "Red"}, False, True
    if(channel_id not in range(100)):
        return "no channel id selected", {"color": "Red"}, False, True

    if(ff_validated and mag_validated and phase_validated):
        ff = np.array([float(i) for i in ff.split(sep)]) * str_to_unit[unit_ff]
        mag = np.array([float(i) for i in mag.split(sep)]) * str_to_unit[unit_mag]
        phase = np.array([float(i) for i in phase.split(sep)]) * str_to_unit[unit_phase]
        if(len(ff) == len(mag) == len(phase)):
            return "all inputs validated", {"color": "Green"}, True, False
        else:
            return "inputs don't have the same length", {"color": "Red"}, False, True

    return "input fields not validated", {"color": "Red"}, False, True


@app.callback(
    [Output("channel-id", "options"),
     Output("channel-id", "value")],
    [Input("trigger", "value"),
    Input("amp-board-list", "value"),
    Input("S-dropdown", "value"),
    Input("allow-override", "value")
    ]
)
def update_dropdown_channel_ids(n_intervals, amp_name, S_parameter, allow_override_checkbox):
    """
    disable all channels that are already in the database for that amp board and S parameter
    """
    allow_override = False
    if 1 in allow_override_checkbox:
        allow_override = True

    existing_ids = det.db.amp_boards.distinct("channels.id", {"name": amp_name, "channels.S_parameter": S_parameter})
    print(f"existing ids for amp {amp_name} and channels.{S_parameter}: {existing_ids}")
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
             State('frequencies', 'value'),
             State('magnitude', 'value'),
             State('phases', 'value'),
             State('dropdown-frequencies', 'value'),
             State('dropdown-magnitude', 'value'),
             State('dropdown-phase', 'value'),
             State("channel-id", "value"),
             State('separator', 'value'),
             State('S-dropdown', 'value')])
def insert_to_db(n_clicks, board_dropdown, new_board_name, ff, mag, phase, unit_ff, unit_mag, unit_phase, channel_id, sep, S_parameter):
    print("insert to db")
    ff = np.array([float(i) for i in ff.split(sep)]) * str_to_unit[unit_ff]
    mag = np.array([float(i) for i in mag.split(sep)]) * str_to_unit[unit_mag]
    phase = np.array([float(i) for i in phase.split(sep)]) * str_to_unit[unit_phase]

    board_name = board_dropdown
    if(board_dropdown == "new"):
        board_name = new_board_name
    print(board_name, channel_id, ff, mag, phase)
    det.insert_amp_board_channel_S12(board_name, S_parameter, channel_id, ff, mag, phase)

    return "/apps/menu"


@app.callback(
    Output('figure-amp', 'figure'),
    [Input("validation-frequencies-output", "data-validated"),
     Input("validation-magnitude-output", "data-validated"),
     Input("validation-phase-output", "data-validated")],
    [State('frequencies', 'value'),
     State('magnitude', 'value'),
     State('phases', 'value'),
     State('dropdown-frequencies', 'value'),
     State('dropdown-magnitude', 'value'),
     State('dropdown-phase', 'value'),
     State('separator', 'value')])
def display_value(val_ff, val_mag, val_phase, ff, mag, phase, unit_ff, unit_mag, unit_phase, sep):
    print("display_value")
    if(val_ff and val_mag and val_phase):
        ff = np.array([float(i) for i in ff.split(sep)]) * str_to_unit[unit_ff]
        mag = np.array([float(i) for i in mag.split(sep)]) * str_to_unit[unit_mag]
        phase = np.array([float(i) for i in phase.split(sep)]) * str_to_unit[unit_phase]
        print(unit_ff)
        fig = subplots.make_subplots(rows=1, cols=2)
        fig.append_trace(go.Scatter(
                    x=ff / units.MHz,
                    y=mag / units.V,
                    opacity=0.7,
                    marker={
                        'color': "blue",
                        'line': {'color': "blue"}
                    },
                    name='magnitude'
                ), 1, 1)
        fig.append_trace(go.Scatter(
                    x=ff / units.MHz,
                    y=phase / units.deg,
                    opacity=0.7,
                    marker={
                        'color': "blue",
                        'line': {'color': "blue"}
                    },
                    name='phase'
                ), 1, 2)
        fig['layout']['xaxis1'].update(title='frequency [MHz]')
        fig['layout']['yaxis1'].update(title='magnitude [V]')
        fig['layout']['yaxis2'].update(title='phase [deg]')
        fig['layout']['xaxis2'].update(title='frequency [MHz]')
        return fig
    else:
        return {"data": []}
