import six
import NuRadioReco.detector.detector
import NuRadioReco.detector.generic_detector
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from app import app
import glob
import astropy.time

@six.add_metaclass(NuRadioReco.detector.detector.Singleton)
class DetectorProvider(object):
    def set_detector(self, filename):
        self.__detector = NuRadioReco.detector.detector.Detector(source='json', json_filename=filename)
        self.__detector.update(astropy.time.Time('2020-1-1'))

    def set_generic_detector(self, filename, default_station, default_channel):
        self.__detector = NuRadioReco.detector.generic_detector.GenericDetector(filename, default_station, default_channel)

    def get_detector(self):
        return self.__detector

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
            ], className='input-group')

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
    [Input('load-detector-button', 'n_clicks')],
    [State('detector-file-dropdown', 'value'),
    State('file-type-dropdown', 'value')])
def open_detector(n_clicks, filename, detector_type):
    print('click')
    if filename is None:
        return ''
    detector_provider = DetectorProvider()
    if detector_type == 'detector':
        detector_provider.set_detector(filename)
    elif detector_type == 'generic_detector':
        detector_provider.set_generic_detector(filename, 101,1)
    return ''
