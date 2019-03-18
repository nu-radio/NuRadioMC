from __future__ import absolute_import, division, print_function  # , unicode_literals
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash
import plotly.graph_objs as go
import json
import numpy as np
import uuid
import glob
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from flask import Flask, send_from_directory
from app import app
from apps import overview
from apps import traces
from apps import cosmic_rays
from apps.common import get_point_index
import apps.simulation
import os
import sys
# from apps import summary
import dataprovider
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('index')

data_folder = os.path.dirname(sys.argv[1])

# Loading screen CSS
# app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})
# app.css.append_css({
#     'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
# })
app.css.append_css({"external_url": "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"})
provider = dataprovider.DataProvider()


app.title = 'ARIANNA viewer'

app.layout = html.Div([
    # represents the URL bar, doesn't render anything
    dcc.Location(id='url', refresh=False),
    html.Div(id='event-click-coordinator', children=json.dumps(None), style={'display': 'none'}),
    html.Div(id='user_id', style={'display': 'none'},
             children=json.dumps(None)),
#     html.Div(id='filename', style={'display': 'none'},
#              children=json.dumps(None)),
    html.Div(id='event-ids',  style={'display': 'none'},
             children=json.dumps([])),
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                html.Div('File location:', className='input-group-text')
                ], className='input-group-addon'),
                dcc.Input(id='datafolder', placeholder='filename', type='text', value=data_folder, className='form-control')
                ], className='input-group'),
            html.Div([
                dcc.Dropdown(id='filename',
                             options=[{'label': l, 'value': l} for l in sorted(glob.glob(data_folder + '/*.ar*'))],
                             className='custom-dropdown'),
                 html.Div([
                    html.Button('open file', id='btn-open-file', className='btn btn-default')
                    ], className='input-group-btn'),
            ], className='input-group'),
            html.Div([
                html.Div([
                        html.Button([
                                html.Div(className='fa fa-arrow-left')
                            ],
                            id='btn-previous-event',
                            className='btn btn-primary',
                            n_clicks_timestamp = 0
                        ),
                        html.Button(
                            id = 'event-number-display',
                            children = '''''',
                            className='btn btn-primary'
                        ),
                        html.Button([
                                html.Div(className='fa fa-arrow-right')
                            ],
                            id='btn-next-event',
                            className='btn btn-primary',
                            n_clicks_timestamp=0
                            )
                    ],
                    className='btn-group',
                    style={'margin': '10px'}
                    ),
                html.Div([
                     dcc.Slider(
                        id='event-counter-slider',
                        step=1,
                        value=0,
                        marks={}
                    ),
                ],
                style={
                'padding': '10px 30px 20px',
                'overflow': 'hidden',
                'flex': '1'}
                ),
                html.Div([
                    dcc.Dropdown(
                        id='station-id-dropdown',
                        options = [],
                        multi=False
                    )
                ],
                style={'flex': 'none', 'padding': '10px', 'min-width': '200px'})
            ],
            style={
            'display': 'flex'
            })
        ], style={'flex': '7'}),
        html.Div([
            html.Div([
                html.Div('Run:', className='custom-table-td'),
                html.Div('', className='custom-table-td-last', id='event-info-run')
            ], className='custom-table-row'),
            html.Div([
                html.Div('Event:', className='custom-table-td'),
                html.Div('', className='custom-table-td-last', id='event-info-id')
            ], className='custom-table-row'),
            html.Div([
                html.Div('Time:', className='custom-table-td'),
                html.Div('', className='custom-table-td-last', id='event-info-time')
            ], className='custom-table-row')
        ], style={'flex': '1'}, className='event-info-table')
    ], style={'display': 'flex'}),
    dcc.Tabs([
        dcc.Tab([
            overview.layout
        ], id='summary-tab', label='Summary'),
        dcc.Tab([
            traces.layout
        ], label='Traces'),
        dcc.Tab([
            apps.simulation.layout
        ], label='Simulation'),
        dcc.Tab([
            cosmic_rays.layout
        ], label='Cosmic Rays')
    ])
])


# next/previous buttons
@app.callback(
Output('event-counter-slider', 'value'),
[Input('btn-next-event', 'n_clicks_timestamp'),
Input('btn-previous-event', 'n_clicks_timestamp'),
Input('event-click-coordinator', 'children')],
[State('event-counter-slider', 'value'),
State('user_id', 'children'),
State('filename', 'value')]
)
def set_event_number(next_evt_click_timestamp, prev_evt_click_timestamp, j_plot_click_info, i_event, juser_id, filename):
    if filename is None:
        return 0
    plot_click_info = json.loads(j_plot_click_info)
    if plot_click_info is not None and plot_click_info['time']*1000. > prev_evt_click_timestamp and plot_click_info['time']*1000. > next_evt_click_timestamp:
        return plot_click_info['event_i']
    else:
        if prev_evt_click_timestamp == 0 and next_evt_click_timestamp == 0:
            return 0
        if prev_evt_click_timestamp > next_evt_click_timestamp:
            if i_event == 0:
                return 0
            else:
                return i_event - 1
        else:
            user_id = json.loads(juser_id)
            
            number_of_events = provider.get_arianna_io(user_id, filename).get_n_events()    
            if number_of_events == i_event + 1:
                return number_of_events -1
            else:
                return i_event + 1


@app.callback(
Output('event-number-display', 'children'),
[Input('filename', 'value'),
Input('event-counter-slider', 'value')]
)
def set_event_number_display(filename, event_number):
    if filename is None:
        return 'No file selected'
    return 'Event {}'.format(event_number)

# slider functions
###############
# set maximum value of slider
@app.callback(
    Output('event-counter-slider', 'max'),
    [Input('filename', 'value')],
    [State('user_id', 'children')])
def update_slider_options(filename, juser_id):
    if filename is None:
        return 0
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    number_of_events = ariio.get_n_events()
    return number_of_events - 1

@app.callback(
Output('event-counter-slider', 'marks'),
[Input('filename', 'value')],
[State('user_id', 'children')]
)
def update_slider_marks(filename, juser_id):
    if filename is None:
        return {}
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    n_events = ariio.get_n_events()
    step_size = int(np.power(10., int(np.log10(n_events))))
    marks = {}
    for i in range(0, n_events, step_size):
        marks[i] = str(i)
    if n_events%step_size != 0:
        marks[n_events] = str(n_events)
    return marks
    

@app.callback(Output('user_id', 'children'),
              [Input('url', 'pathname')],
              [State('user_id', 'children')])
def set_uuid(pathname, juser_id):
    user_id = json.loads(juser_id)
    if(user_id is None):
        user_id = uuid.uuid4().hex
    return json.dumps(user_id)


@app.callback(Output('filename', 'options'),
              [Input('datafolder', 'value')])
def set_filename_dropdown(folder):
    return [{'label': l, 'value': l} for l in sorted(glob.glob(os.path.join(folder, '*.nur*')))]



@app.callback(Output('station-id-dropdown', 'options'),
            [Input('filename', 'value'),
            Input('event-counter-slider', 'value')],
            [State('user_id', 'children')])
def get_station_dropdown_options(filename, i_event, juser_id):
    if filename is None:
        return []
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    event = ariio.get_event_i(i_event)
    dropdown_options = []
    for station in event.get_stations():
        dropdown_options.append({
            'label': 'Station {}'.format(station.get_id()),
            'value': station.get_id()
        })
    return dropdown_options

@app.callback(Output('station-id-dropdown', 'value'),
            [Input('filename', 'value'),
            Input('event-counter-slider', 'value')],
            [State('user_id', 'children')])
def set_to_first_station_in_event(filename, event_i, juser_id):
    if filename is None:
        return None
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    event = ariio.get_event_i(event_i)
    for station in event.get_stations():
        return station.get_id()

@app.callback(Output('summary', 'style'),
              [Input('url', 'pathname')])
def display_page2(pathname):
    if pathname == '/apps/traces':
        return {'display': 'none'}
    if pathname == '/apps/summary':
        return {}
#     elif pathname == '/apps/app2':
#         return app2.layout
    else:
        return '404'


@app.callback(Output('skyplot-xcorr', 'figure'),
              [Input('filename', 'value'),
               Input('trigger', 'children'),
               Input('event-ids', 'children'),
               Input('station-id-dropdown', 'value')],
              [State('user_id', 'children')])
def plot_skyplot_xcorr(filename, trigger, jcurrent_selection, station_id, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    current_selection = json.loads(jcurrent_selection)
    ariio = provider.get_arianna_io(user_id, filename)
    traces = []
    keys = ariio.get_header()[station_id].keys()
    if stnp.zenith in keys and stnp.azimuth in keys:
        traces.append(go.Scatterpolar(
            r=np.rad2deg(ariio.get_header()[station_id][stnp.zenith]),
            theta=np.rad2deg(ariio.get_header()[station_id][stnp.azimuth]),
            text=[str(x) for x in ariio.get_event_ids()],
            mode='markers',
            name='all events',
            opacity=1,
            marker=dict(
                color='blue'
            )
        ))
    else:
        return {}

    # update with current selection
    if current_selection != []:
        for trace in traces:
            trace['selectedpoints'] = current_selection

    return {
        'data': traces,
        'layout': go.Layout(
            showlegend= True,
#             xaxis={'type': 'linear', 'title': ''},
#             yaxis={'title': xcorr_states[xcorr_type], 'range': [0, 1]},
#             margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
#             legend={'x': 0, 'y': 1},
            hovermode='closest',
            height=500
        )
    }


# update event ids list from plot selection
@app.callback(Output('event-ids', 'children'),
              [Input('cr-skyplot', 'selectedData'),
               Input('cr-xcorrelation', 'selectedData'),
               Input('cr-xcorrelation-amplitude', 'selectedData'),
               Input('skyplot-xcorr', 'selectedData'),
               Input('cr-polarization-zenith', 'selectedData')],
              [State('event-ids', 'children')])
def set_event_selection(selectedData1, selectedData2, selectedData3, selectedData4, selectedData5, jcurrent_selection):
    current_selection = json.loads(jcurrent_selection)
    tcurrent_selection = []
    #print(selectedData3)
    for i, selection in enumerate([selectedData1, selectedData2, selectedData3, selectedData4, selectedData5]):  # check which selection has fired the callback
        if selection is not None:
            event_ids = []
            for x in selection['points']:
                t = x['customdata']
                if t not in event_ids:
                    event_ids.append(t)
            if not np.array_equal(np.array(event_ids), current_selection):  # this selection has fired the callback
                tcurrent_selection = event_ids
    return json.dumps(tcurrent_selection)

def add_click_info(json_object, event_number_array, times_array):
    object = json.loads(json_object)
    if object is not None:
        event_number_array.append(object['event_i'])
        times_array.append(object['time'])

        #finds out which one of the plots was clicked last (i.e. which one triggered the event update)
@app.callback(Output('event-click-coordinator', 'children'),
            [Input('cr-polarization-zenith-point-click', 'children'),
            Input('cr-skyplot-point-click', 'children'),
            Input('cr-xcorrelation-point-click', 'children'),
            Input('cr-xcorrelation-amplitude-point-click', 'children')])
def coordinate_event_click(cr_polarization_zenith_click, cr_skyplot_click, cr_xcorrelation_click, cr_xcorrelation_amplitude_click):
    event_numbers = []
    times = []
    add_click_info(cr_polarization_zenith_click, event_numbers, times)
    add_click_info(cr_skyplot_click, event_numbers, times)
    add_click_info(cr_xcorrelation_click, event_numbers, times)
    add_click_info(cr_xcorrelation_amplitude_click, event_numbers, times)
    if len(times) == 0:
        return json.dumps(None)
    i = np.argmax(times)
    return json.dumps({
        'event_i': event_numbers[i],
        'time': times[i]
    })

@app.callback(Output('event-info-run', 'children'),
            [Input('event-counter-slider', 'value'),
            Input('filename', 'value')],
            [State('user_id', 'children')])
def update_event_info_run(event_i, filename, juser_id):
    if filename is None:
        return ""
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(event_i)
    return evt.get_run_number()    

@app.callback(Output('event-info-id', 'children'),
            [Input('event-counter-slider', 'value'),
            Input('filename', 'value')],
            [State('user_id', 'children')])
def update_event_info_id(event_i, filename, juser_id):
    if filename is None:
        return ""
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(event_i)
    return evt.get_id()

@app.callback(Output('event-info-time', 'children'),
            [Input('event-counter-slider', 'value'),
            Input('filename', 'value'),
            Input('station-id-dropdown', 'value')],
            [State('user_id', 'children')])
def update_event_info_time(event_i, filename, station_id, juser_id):
    if filename is None or station_id is None:
        return ""
#     filename = json.loads(jfilename)
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(event_i)
    if evt.get_station(station_id).get_station_time() is None:
        return ''
    return '{:%d. %b %Y, %H:%M:%S}'.format(evt.get_station(station_id).get_station_time())

if __name__ == '__main__':
    if int(dash.__version__.split('.')[1]) < 39:
        print('WARNING: Dash version 0.39.0 or newer is required, you are running version {}. Please update.'.format(dash.__version__))
    app.run_server(debug=True, port=8080)
