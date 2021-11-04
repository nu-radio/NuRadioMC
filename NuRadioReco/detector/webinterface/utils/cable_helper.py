import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from NuRadioReco.utilities import units
from plotly import subplots
import numpy as np
import plotly.graph_objs as go
import sys
import base64
from io import StringIO
from NuRadioReco.detector.webinterface.app import app
from NuRadioReco.detector import detector_mongo as det
from NuRadioReco.detector.webinterface.utils.table import get_table
from NuRadioReco.detector.webinterface.utils.units import str_to_unit

sparameters_layout = html.Div([html.Br(),
    dcc.Checklist(id="function-test",
        options=[
            {'label': 'Channel is working', 'value': 'working'},
            {'label': 'Is this the Primary Measurement?', 'value': 'primary'}
            ],
        value=['working', 'primary'],
        style={'width': '10%'}
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
    html.Div([html.Div("units"),
    dcc.Dropdown(
        id='dropdown-frequencies',
        options=[
            {'label': 'GHz', 'value': "GHz"},
            {'label': 'MHz', 'value': "MHz"},
            {'label': 'Hz', 'value': "Hz"}
        ],
        value="Hz",
        style={'width': '100px', 'float':'left'}
    ),
    dcc.Dropdown(
            id='dropdown-magnitude',
            options=[
                {'label': 'dB', 'value': "dB"},
                {'label': 'MAG', 'value': "MAG"}
            ],
            value="dB",
            style={'width': '100px', 'float':'left'}
        ),
    dcc.Dropdown(
        id='dropdown-phase',
        options=[
            {'label': 'degree', 'value': "deg"},
            {'label': 'rad', 'value': "rad"}
        ],
        value="deg",
        style={'width': '100px', 'float':'left'}
    ), ]),
    html.Br(),
    html.Br(),
    html.Div("Specify the measurement protocol"),
    dcc.Dropdown(id="protocol",
                options=[
                    {'label': 'Chicago 2020', 'value': "chicago2020"},
                    {'label': 'Erlangen 2020', 'value': "erlangen2020"}],
              value="chicago2020",
              style={'width': '200px',
                     'float': 'left'}),
    html.Br(),
    html.Br(),
    html.Div([
        dcc.Upload(
        id='S21_mag_data',
        children=html.Div([
            'Drag and Drop your Mag CSV or ',
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
    html.Div([
        dcc.Upload(
        id='S21_phase_data',
        children=html.Div([
            'Drag and Drop your Phase CSV or ',
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
    html.Div('', id='validation-S21data-output', style={'whiteSpace': 'pre-wrap'})])


# @app.callback(
#     Output('new-amp-cable-input', 'disabled'),
#     [Input('amp-cable-list', 'value')])
# def enable_cable_name_input(value):
#     if(value == "new"):
#         return False
#     else:
#         return True
#
#
# @app.callback(
#     Output("amp-cable-list", "options"),
#     [Input("trigger", "children")],
#     [State("amp-cable-list", "options"),
#      State("table-name", "children")]
# )
# def update_dropdown_cable_names(n_intervals, options, table_name):
#     """
#     updates the dropdown menu with existing cable names from the database
#     """
#     for amp_name in get_table(table_name).distinct("name"):
#         options.append(
#             {"label": amp_name, "value": amp_name}
#         )
#     print(f"update_dropdown_amp_names = {options}")
#     return options


@app.callback(
    [Output("validation-S21data-output", "children"),
     Output("validation-S21data-output", "style"),
    Output("validation-S21data-output", "data-validated")],
    [Input('S21_mag_data', 'contents'),
     Input('S21_phase_data', 'contents'),
     Input('dropdown-frequencies', 'value'),
     Input('dropdown-magnitude', 'value'),
     Input('dropdown-phase', 'value'),
     Input('separator', 'value')
     ])
def validate_Sdata(S21_mag_data, S21_phase_data, unit_ff, unit_mag, unit_phase, sep):
    """
    validates frequency array

    displays string with validation information for the user

    The outcome of the validataion (True/False) is saved in the 'data-validated' attribute to trigger other actions.
    """
    if((S21_mag_data != "") and (S21_phase_data != "")):
        try:
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
            tmp = [f"you entered {len(Sm_data[0])} frequencies from {Sm_data[0].min()/units.MHz:.4g} MHz to {Sm_data[0].max()/units.MHz:.4g} MHz"]
            tmp.append(html.Br())
            tmp.append(f"{len(Sm_data[1])} S21 magnitudes from {Sm_data[1].min():.4g} dB to {Sm_data[1].max():.4g} dB")
            tmp.append(html.Br())
            tmp.append(f"{len(Sp_data[1])} S21 phase measurements from {Sp_data[1].min():.4g} rad to {Sp_data[1].max():.4g} rad")


            return tmp, {"color": "Green"}, True
        except:
    #         print(sys.exc_info())
            return f"{sys.exc_info()[0].__name__}:{sys.exc_info()[1]}", {"color": "Red"}, False
    else:
        return f"no data inserted", {"color": "Red"}, False


@app.callback(
    Output('figure-cable', 'figure'),
    [Input("validation-S21data-output", "data-validated")],
    [State('S21_mag_data', 'contents'),
             State('S21_phase_data', 'contents'),
             State('dropdown-frequencies', 'value'),
             State('dropdown-magnitude', 'value'),
             State('dropdown-phase', 'value'),
             State('separator', 'value')])
def plot_Sparameters(val_Sdata, S21_mag_data, S21_phase_data, unit_ff, unit_mag, unit_phase, sep):
    print("display_value")
    if(val_Sdata):
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

            fig = subplots.make_subplots(rows=1, cols=2)
            fig.append_trace(go.Scatter(
                            x=Sm_data[0] /units.MHz,
                            y=Sm_data[1],
                            opacity=0.7,
                            marker={
                                'color': "blue",
                                'line': {'color': "blue"}
                            },
                            name='magnitude'
                        ), 1, 1)
            fig.append_trace(go.Scatter(
                            x=Sm_data[0] /units.MHz,
                            y=Sp_data[1] /units.deg,
                            opacity=0.7,
                            marker={
                                'color': "red",
                                'line': {'color': "red"}
                            },
                            name='phase'
                        ), 1, 2)
            fig['layout']['xaxis1'].update(title='frequency [MHz]')
            fig['layout']['yaxis1'].update(title='mag[dB]')
            fig['layout']['xaxis2'].update(title='frequency [MHz]')
            fig['layout']['yaxis2'].update(title='phase [degrees]')
            return fig
    else:
            return {"data": []}
