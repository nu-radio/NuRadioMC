import six
import NuRadioReco.detector.detector
import NuRadioReco.detector.generic_detector
import NuRadioReco.modules.io.NuRadioRecoio
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from app import app
import glob
import astropy.time
import numpy as np


@six.add_metaclass(NuRadioReco.detector.detector.Singleton)
class DetectorProvider(object):
    def __init__(self):
        self.__detector = None
        self.__io = None
    def set_detector(self, filename):
        self.__detector = NuRadioReco.detector.detector.Detector.__new__(NuRadioReco.detector.detector.Detector)
        self.__detector.__init__(source='json', json_filename=filename)

    def set_generic_detector(self, filename, default_station, default_channel):
        self.__detector = NuRadioReco.detector.generic_detector.GenericDetector.__new__(NuRadioReco.detector.generic_detector.GenericDetector)
        self.__detector.__init__(filename, default_station, default_channel)

    def set_event_file(self, filename):
        self.__io = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio([filename])
        event = self.__io.get_event_i(0)
        self.__detector = self.__io.get_detector()
        for station in event.get_stations():
            self.__detector.update(station.get_station_time())
            break

    def get_detector(self):
        return self.__detector

    def set_time_periods(self, unix, astropy_time):
        self.__unix_time_periods = unix
        self.__astropy_time_periods = astropy_time

    def get_time_periods(self):
        return self.__unix_time_periods, self.__astropy_time_periods

    def get_n_events(self):
        if self.__io is None:
            return 0
        return self.__io.get_n_events()

    def get_event_ids(self):
        if self.__io is None:
            return None
        return self.__io.get_event_ids()


layout = html.Div([
    html.Div([
        html.Div('', id='output-dummy', style={'display': 'none'}),
        html.Div('File Selection', className='panel panel-heading'),
        html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='file-type-dropdown',
                        options=[
                            {'label': 'Detector', 'value':'detector'},
                            {'label': 'Generic Detector', 'value':'generic_detector'},
                            {'label': 'Event File', 'value':'event_file'}
                        ],
                        value='detector',
                        multi=False
                    )
                ], style={'flex': '1'}),
                html.Div([
                    dcc.Input(
                        id='folder-name-input',
                        type='text',
                        value='Test',
                        className='form-control'
                    )
                ], style={'flex': '1'}),
                html.Button(
                    'Refresh',
                    id='folder-name-refresh',
                    n_clicks=0,
                    className='btn btn-primary'
                )
            ], className='input-group'),
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='detector-file-dropdown',
                        options=[],
                        value=None,
                        multi=False
                    )
                ], style={'flex': '1'}),
                html.Button(
                    'Open',
                    id='load-detector-button',
                    n_clicks=0,
                    className='btn btn-primary'
                )
            ], className='input-group'),
            html.Div([
                dcc.Input(
                    id='default-station-input',
                    type='number',
                    value=None,
                    placeholder='Default Station',
                    className='form-control'
                ),
                dcc.Input(
                    id='default-channel-input',
                    type='number',
                    value=None,
                    placeholder='Default Channel',
                    className='form-control'
                )
            ],id='default-settings-div', className='input-group'),
            html.Div([
                html.Button(
                    '',
                    id='selected-data-button',
                    className='badge badge-light'
                ),
                html.Div([
                    dcc.Slider(
                        id='detector-time-slider',
                        value=10000,
                        min=0,
                        max=1551092200,
                        step=1000
                    )
                ], style={'flex': '1'}),
                html.Button(
                'Update',
                id='update-detector-time-button',
                n_clicks=0,
                className='btn btn-primary'
                )
            ], id='detector-time-div', className='input-group'),
            html.Div([
                html.Button(
                    '     ',
                    id='selected-event-button',
                    className='badge badge-light'
                ),
                html.Div([
                    dcc.Slider(
                        id='detector-event-slider',
                        value=0,
                        min=0,
                        max=1,
                        step=1
                    )
                ], style={'flex': '1'}),
                html.Button(
                    'Update',
                    id='update-detector-event-button',
                    n_clicks=0,
                    className='btn btn-primary'
                )
            ], id='detector-event-div', className='input-group')
        ], className='panel panel-body')
    ], className='panel panel-default')
])


@app.callback(
    Output('folder-name-input', 'value'),
    [Input('folder-dummy', 'children')]
)
def set_folder_name(name):
    return name

@app.callback(
    Output('detector-file-dropdown', 'options'),
    [Input('folder-dummy', 'children'),
    Input('folder-name-refresh', 'n_clicks'),
    Input('file-type-dropdown', 'value')],
    [State('folder-name-input', 'value')]
)
def update_file_name_options(folder_dummy, refresh_button, file_type, folder_input):
    context = dash.callback_context
    options = []

    if file_type == 'event_file':
        suffix = '/*.nur'
    else:
        suffix = '/*.json'
    if context.triggered[0]['prop_id'] == 'folder-dummy.children':
        for filename in glob.glob(folder_dummy+suffix):
            options.append({'label': filename, 'value': filename})
    else:
        for filename in glob.glob(folder_input+suffix):
            options.append({'label': filename, 'value': filename})
    return options

@app.callback(Output('output-dummy', 'children'),
    [Input('load-detector-button', 'n_clicks'),
    Input('update-detector-time-button', 'n_clicks')],
    [State('detector-file-dropdown', 'value'),
    State('file-type-dropdown', 'value'),
    State('default-station-input', 'value'),
    State('default-channel-input', 'value'),
    State('detector-time-slider', 'value')])
def open_detector(n_clicks, time_n_clicks, filename, detector_type, default_station, default_channel, detector_time):
    if filename is None:
        return ''
    detector_provider = DetectorProvider()
    context = dash.callback_context
    if context.triggered[0]['prop_id'] == 'update-detector-time-button.n_clicks':
        detector_provider.get_detector().update(astropy.time.Time(detector_time, format='unix'))
        return n_clicks
    if detector_type == 'detector':
        detector_provider.set_detector(filename)
        detector = detector_provider.get_detector()
        unix_times = []
        datetimes = []
        for station_id in detector.get_station_ids():
            for dt in detector.get_unique_time_periods(station_id):
                if dt.unix not in unix_times:
                    unix_times.append(dt.unix)
                    datetimes.append(dt)
        detector_provider.set_time_periods(unix_times, datetimes)
        detector.update(datetimes[np.argmin(unix_times)])
    elif detector_type == 'generic_detector':
        detector_provider.set_generic_detector(filename, default_station,default_channel)
    elif detector_type == 'event_file':
        detector_provider.set_event_file(filename)
    return n_clicks

@app.callback(
    Output('default-settings-div', 'style'),
    [Input('file-type-dropdown', 'value')]
)
def show_default_settings_div(detector_type):
    if detector_type == 'generic_detector':
        return {'z-index': '0'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('load-detector-button', 'disabled'),
    [Input('detector-file-dropdown', 'value'),
    Input('file-type-dropdown', 'value'),
    Input('default-station-input', 'value')]
)
def toggle_open_button_active(filename, detector_type, default_station):
    if filename is None:
        return True
    if detector_type == 'generic_detector' and default_station is None:
        return True
    return False

@app.callback(
    Output('detector-time-div', 'style'),
    [Input('load-detector-button', 'n_clicks'),
    Input('file-type-dropdown', 'value')]
)
def show_detector_time_slider(load_detector_click,detector_type):
    if detector_type == 'detector':
        detector_provider = DetectorProvider()
        if detector_provider.get_detector() is not None:
            return {'z-index': '0'}
    return {'display': 'none'}

@app.callback(
    [Output('detector-time-slider', 'value'),
    Output('detector-time-slider', 'min'),
    Output('detector-time-slider', 'max'),
    Output('detector-time-slider', 'marks')],
    [Input('output-dummy', 'children'),
    Input('file-type-dropdown', 'value')]
)
def set_detector_time_slider(load_detector_click, detector_type):
    detector_provider = DetectorProvider()
    detector = detector_provider.get_detector()
    if detector is None:
        return None, 0, 1, {}
    if detector_type != 'detector':
        return None, 0, 1, {}
    unix_times, datetimes = detector_provider.get_time_periods()
    marks = {}
    for i_time, unix_time in enumerate(unix_times):
        datetimes[i_time].format = 'iso'
        marks[unix_time] = {'label': str(datetimes[i_time].value)}
    return detector.get_detector_time().unix, np.min(unix_times), np.max(unix_times), marks

@app.callback(
    Output('selected-data-button', 'children'),
    [Input('detector-time-slider', 'value')]
)
def set_selected_date_button(unix_time):
    if unix_time is None:
        return ''
    t = astropy.time.Time(unix_time, format='unix')
    t.format = 'iso'
    return str(t.value).split(' ')[0]

@app.callback(
    Output('detector-event-div', 'style'),
    [Input('file-type-dropdown', 'value')]
)
def show_event_selection(file_type):
    if file_type == 'event_file':
        return {}
    else:
        return {'display': 'none'}

@app.callback(
    [Output('detector-event-slider', 'value'),
    Output('detector-event-slider', 'max')],
    [Input('output-dummy', 'children'),
    Input('file-type-dropdown', 'value')]
)
def set_detector_event_slider(load_detector_click, detector_type):
    detector_provider = DetectorProvider()
    detector = detector_provider.get_detector()
    if detector is None:
        return 0, 0
    if detector_type != 'event_file':
        return 0, 0
    return 0, detector_provider.get_n_events()

@app.callback(
    Output('selected-event-button', 'children'),
    [Input('detector-event-slider', 'value')]
)
def set_event_id_display(i_event):
    detector_provider = DetectorProvider()
    event_ids = detector_provider.get_event_ids()
    if event_ids is None:
        return '     '
    return 'Run {}, Event {}'.format(event_ids[i_event][0], event_ids[i_event][1])
