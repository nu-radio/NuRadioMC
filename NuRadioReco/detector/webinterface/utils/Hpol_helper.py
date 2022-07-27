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
from NuRadioReco.detector.webinterface.app import app
from NuRadioReco.detector.webinterface.utils.table import get_table
from NuRadioReco.detector.webinterface.utils.units import str_to_unit

helper_name = 'HPol'

sparameters_layout = html.Div([
    dcc.Checklist(id="function-test",
        options=[
            {'label': 'channel is working', 'value': 'working'},
            {'label': 'Is this the Primary Measurement?', 'value': 'primary'}
            ],
        value=['working', 'primary'],
        style={'width': '20%'}
    ), html.Br(),

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
    html.Br(),
    html.Br(),
    html.Div([html.Div("Units"),
    dcc.Dropdown(
        id='dropdown-frequencies',
        options=[
            {'label': 'GHz', 'value': "GHz"},
            {'label': 'MHz', 'value': "MHz"},
            {'label': 'Hz', 'value': "Hz"}
        ],
        value="Hz",
        style={'width': '100px', 'float': 'left'}
    ),
    dcc.Dropdown(
            id='dropdown-magnitude',
            options=[
                {'label': 'dB', 'value': "dB"},
                {'label': 'V', 'value': "V"},
                {'label': 'mV', 'value': "mV"}
            ],
            value="dB",
            style={'width': '100px', 'float': 'left'}
        ), ]),
    html.Br(),
    html.Br(),
    html.Div('Specify the measurement protocol'),
    dcc.Dropdown(id='protocol',
                 options=[{'label': 'PennState 2021', 'value': 'pennstate2021'},
                          {'label': 'PennState 2022', 'value': 'pennstate2022'}],
                 value='pennstate2021',
                 style={'width':'200px', 'float':'left'}),
    html.Br(),
    html.Br(),
    html.Div([
        dcc.Upload(
        id='Sdata',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    ], style={'width':'100%', 'float': 'hidden'}),
    html.Br(),
    html.Div('', id=helper_name+'-validation-Sdata-output', style={'whiteSpace': 'pre-wrap'})])


# @app.callback(
#     Output('new-board-input', 'disabled'),
#     [Input('amp-board-list', 'value')])
# def enable_board_name_input(value):
#     if(value == "new"):
#         return False
#     else:
#         return True


# @app.callback(
#     Output("amp-board-list", "options"),
#     [Input("trigger", "children")],
#     [State("amp-board-list", "options"),
#      State("table-name", "children")]
# )
# def update_dropdown_amp_names(n_intervals, options, table_name):
#     """
#     updates the dropdown menu with existing board names from the database
#     """
#     for amp_name in get_table(table_name).distinct("name"):
#         options.append(
#             {"label": amp_name, "value": amp_name}
#         )
#     print(f"update_dropdown_amp_names = {options}")
#     return options


@app.callback(
    [Output(helper_name+"-validation-Sdata-output", "children"),
     Output(helper_name+"-validation-Sdata-output", "style"),
    Output(helper_name+"-validation-Sdata-output", "data-validated")],
    [Input('Sdata', 'contents'),
     Input('dropdown-frequencies', 'value'),
     Input('dropdown-magnitude', 'value'),
     Input('separator', 'value')
     ])
def validate_Sdata(contents, unit_ff, unit_mag, sep):
    """
    validates frequency array

    displays string with validation information for the user

    The outcome of the validataion (True/False) is saved in the 'data-validated' attribute to trigger other actions.
    """
    if(contents != ""):
        try:
            content_type, content_string = contents.split(',')
            S_data = base64.b64decode(content_string)
            S_data_io = StringIO(S_data.decode('utf-8'))
            S_data = np.genfromtxt(S_data_io, skip_header=30, skip_footer=1, delimiter=sep).T
            S_data[0] *= str_to_unit[unit_ff]
            S_data[1] *= str_to_unit[unit_mag]
            tmp = [f"you entered {len(S_data[0])} frequencies from {S_data[0].min()/units.MHz:.4g}MHz to {S_data[0].max()/units.MHz:.4g}MHz"]
            tmp.append(html.Br())
            S_names = "S11"
            tmp.append(f"{S_names} mag {len(S_data[1])} values within the range of {S_data[1].min()/str_to_unit[unit_mag]:.4g}{unit_mag} to {S_data[1].max()/str_to_unit[unit_mag]:.4g}{unit_mag}")

            return tmp, {"color": "Green"}, True
        except:
    #         print(sys.exc_info())
            return f"{sys.exc_info()[0].__name__}:{sys.exc_info()[1]}", {"color": "Red"}, False
    else:
        return f"no data inserted", {"color": "Red"}, False


@app.callback(
    Output('figure-HPol', 'figure'),
    [Input(helper_name+"-validation-Sdata-output", "data-validated")],
    [State('Sdata', 'contents'),
             State('dropdown-frequencies', 'value'),
             State('dropdown-magnitude', 'value'),
             State('separator', 'value')])
def plot_Sparameters(val_Sdata, Sdata, unit_ff, unit_mag, sep):
    print("display_value")
    if(val_Sdata):
        content_type, content_string = Sdata.split(',')
        S_data = base64.b64decode(content_string)
        S_data_io = StringIO(S_data.decode('utf-8'))
        S_data = np.genfromtxt(S_data_io, skip_header=30, skip_footer=1, delimiter=sep).T
        S_data[0] *= str_to_unit[unit_ff]
        S_data[1] *= str_to_unit[unit_mag]

        fig = subplots.make_subplots(rows=1, cols=1)
        fig.append_trace(go.Scatter(
            x=S_data[0] / units.MHz,
            y=S_data[1],
            opacity=0.7,
            marker={
                'color': "blue",
                'line': {'color': "blue"}
            },
            name='magnitude'
        ), 1, 1)
        fig['layout']['xaxis1'].update(title='frequency [MHz]')
        fig['layout']['yaxis1'].update(title=unit_mag)
        return fig
    else:
        return {"data": []}
