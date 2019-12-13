import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from NuRadioReco.utilities import units
from plotly import subplots
import numpy as np
import plotly.graph_objs as go

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
    html.Div([
        dcc.Textarea(
            id='frequencies',
            placeholder='frequencies comma separated',
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
    html.Div([
        dcc.Textarea(
            id='magnitude',
            placeholder='magnitudes comma separated',
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
    html.Div([
        dcc.Textarea(
            id='phases',
            placeholder='phase comma separated',
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
    html.Div(id='app-1-display-value'),
    html.Div([
        html.Button('plot', id='button-plot'),
        html.Button('insert to DB', id='button-insert'),
        dcc.Link('Go to App 2', href='/apps/app2')
    ], style={'width':"100%", "overflow": "hidden"}),
    html.Div(id='dd-output-container'),
    dcc.Graph(id='figure-amp')
])


@app.callback(
    Output("amp-board-list", "options"),
    [Input("trigger", "value")],
    [State("amp-board-list", "options")]
)
def update_dropdown(n_intervals, options):
    for amp_name in det.get_amp_board_names():
        options.append(
            {"label": amp_name, "value": amp_name}
        )
    return options


@app.callback(
    Output("channel-id", "options"),
    [Input("trigger", "value"),
    Input("amp-board-list", "value")]
)
def update_dropdown_channel_ids(n_intervals, amp_name):
    existing_ids = det.db.amp_boards.distinct("channels.id", {"name": amp_name})
    print(f"existing ids for amp {amp_name}: {existing_ids}")
    options = []
    for i in range(8):
        if(i in existing_ids):
            options.append({"label": i, "value": i, 'disabled': True})
        else:
            options.append({"label": i, "value": i})
    return options


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
             State("channel-id", "value")])
def insert_to_db(n_clicks, board_dropdown, new_board_name, ff, mag, phase, unit_ff, unit_mag, unit_phase, channel_id):
    print("insert to db")
    ff = np.array([float(i) for i in ff.split(',')]) * str_to_unit[unit_ff]
    mag = np.array([float(i) for i in mag.split(',')]) * str_to_unit[unit_mag]
    phase = np.array([float(i) for i in phase.split(',')]) * str_to_unit[unit_phase]

    board_name = board_dropdown
    if(board_dropdown == "new"):
        board_name = new_board_name
    print(board_name, channel_id, ff, mag, phase)
    det.insert_amp_board_channel_S12(board_name, channel_id, ff, mag, phase)

    return "/"


@app.callback(
    Output('figure-amp', 'figure'),
    [Input('button-plot', 'n_clicks_timestamp'),
     Input('frequencies', 'value'),
     Input('magnitude', 'value'),
     Input('phases', 'value'),
     Input('dropdown-frequencies', 'value'),
     Input('dropdown-magnitude', 'value'),
     Input('dropdown-phase', 'value')])
def display_value(n_clicks_timestamp, ff, mag, phase, unit_ff, unit_mag, unit_phase):
    print("display_value")
    ff = np.array([float(i) for i in ff.split(',')]) * str_to_unit[unit_ff]
    mag = np.array([float(i) for i in mag.split(',')]) * str_to_unit[unit_mag]
    phase = np.array([float(i) for i in phase.split(',')]) * str_to_unit[unit_phase]
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
