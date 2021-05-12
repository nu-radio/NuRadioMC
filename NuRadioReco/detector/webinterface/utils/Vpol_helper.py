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
from NuRadioReco.detector import detector_mongo as det
from NuRadioReco.detector.webinterface.utils.table import get_table
from NuRadioReco.detector.webinterface.utils.units import str_to_unit

sparameters_layout = html.Div([
    dcc.Checklist(id="function-test",
        options=[
            {'label': 'channel is working', 'value': 'working'}
            ],
        value=['working'],
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
                {'label': 'VSWR', 'value': "VSWR"},
                {'label': 'V', 'value': "V"},
                {'label': 'mV', 'value': "mV"}
            ],
            value="VSWR",
            style={'width': '20%',
#                    'float': 'left'}
                   }

        ),
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
    html.Div('', id='validation-S11data-output', style={'whiteSpace': 'pre-wrap'})])


# @app.callback(
#     Output('new-VPol-input', 'disabled'),
#     [Input('VPol-list', 'value')])
# def enable_VPol_name_input(value):
#     if(value == "new"):
#         return False
#     else:
#         return True


# @app.callback(
#     Output("VPol-list", "options"),
#     [Input("trigger", "children")],
#     [State("VPol-list", "options"),
#      State("table-name", "children")]
# )
# def update_dropdown_VPol_names(n_intervals, options, table_name):
#     """
#     updates the dropdown menu with existing antenna names from the database
#     """
#     for VPol_name in get_table(table_name).distinct("name"):
#         options.append(
#             {"label": VPol_name, "value": VPol_name}
#         )
#     print(f"update_dropdown_VPol_names = {options}")
#     return options


@app.callback(
    [Output("validation-S11data-output", "children"),
     Output("validation-S11data-output", "style"),
    Output("validation-S11data-output", "data-validated")],
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
            S_datas = base64.b64decode(content_string)
            S_data_io = StringIO(S_datas.decode('utf-8'))
            S_data = np.genfromtxt(S_data_io, skip_header=17, skip_footer=1, delimiter=sep).T
            S_data[0] *= str_to_unit[unit_ff]
            S_data[1] *= str_to_unit[unit_mag]
            tmp = [f"you entered {len(S_data[0])} frequencies from {S_data[0].min()/units.MHz:.4g}MHz to {S_data[0].max()/units.MHz:.4g}MHz"]
            tmp.append(html.Br())
            S_names = "S11"
            tmp.append(f"{S_names} mag {len(S_data[1])} values within the range of {S_data[1].min():.4g}VSWR to {S_data[1].max():.4g}VSWR")

            return tmp, {"color": "Green"}, True
        except:
        #    print(sys.exc_info())
            return f"{sys.exc_info()[0].__name__}:{sys.exc_info()[1]}", {"color": "Red"}, False
    else:
        return f"no data inserted", {"color": "Red"}, False


@app.callback(
    Output('figure-VPol', 'figure'),
    [Input("validation-S11data-output", "data-validated")],
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
        S_data = np.genfromtxt(S_data_io, skip_header=17, skip_footer=1, delimiter=sep).T
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
        fig['layout']['yaxis1'].update(title='VSWR')
        return fig
    else:
        return {"data": []}
