import NuRadioReco.detector.detector_browser.detector_provider
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import NuRadioReco.detector.ARIANNA.analog_components
import NuRadioReco.detector.RNO_G.analog_components
from NuRadioReco.detector.detector_browser.app import app
import scipy.signal
from NuRadioReco.utilities import units
import NuRadioReco.detector.antennapattern
antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()

layout = html.Div([
    html.Div([
        html.Div('Hardware Response', className='panel panel-heading'),
        html.Div([
            html.Div([
                html.Div([
                    html.Div('Item', className='option-label'),
                    html.Div([
                        dcc.RadioItems(
                            id='response-type-radio',
                            value='amp+antenna',
                            options=[
                                {'label': 'Amplifier', 'value': 'amp'},
                                {'label': 'Antenna', 'value': 'antenna'},
                                {'label': 'Antenna & Amplifier', 'value': 'amp+antenna'}
                            ]
                        )
                    ], className='option-select')
                ], className='option-set'),
                html.Div([
                    html.Div('Signal Zenith', className='option-label'),
                    html.Div([
                        dcc.Slider(
                            id='signal-zenith',
                            min=0,
                            max=180,
                            step=5,
                            value=90,
                            marks={
                                0: '0°',
                                45: '45°',
                                90: '90°',
                                135: '135°',
                                180: '180°'
                            }
                        )
                    ], className='option-select')
                ], className='option-set'),
                html.Div([
                    html.Div('Signal Azimuth', className='option-label'),
                    html.Div([
                        dcc.Slider(
                            id='signal-azimuth',
                            min=0,
                            max=360,
                            step=10,
                            value=180,
                            marks={
                                0: '0° [E]',
                                90: '90° [S]',
                                180: '180° [W]',
                                270: '270° [N]'
                            }
                        )
                    ], className='option-select')
                ], className='option-set'),
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id='log-checklist',
                            options=[
                                {'label': 'Log', 'value': 'log'},
                                {'label': 'Detrend', 'value': 'detrend'}
                            ],
                            value=[],
                            labelStyle={'display': 'block'}
                        )
                    ], className='option-select')
                ], className='option-set', style={'flex': 'none'})
            ], style={'display': 'flex'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='hardware-response-amplitude')
                ], style={'flex': '1'}),
                html.Div([
                    dcc.Graph(id='hardware-response-phase')
                ], style={'flex': '1'})
            ], style={'display': 'flex'})
        ], className='panel panel-body')
    ], className='panel panel-default')
])


@app.callback(
    [Output('hardware-response-amplitude', 'figure'),
        Output('hardware-response-phase', 'figure')],
    [Input('response-type-radio', 'value'),
        Input('signal-zenith', 'value'),
        Input('signal-azimuth', 'value'),
        Input('selected-station', 'children'),
        Input('selected-channel', 'children'),
        Input('log-checklist', 'value')]
)
def draw_hardware_response(response_type, zenith, azimuth, station_id, channel_id, log_checklist):
    """
    Draws plot for antenna and amplifier response

    Parameters:
    -----------------------
    response_type: array of strings
        Contains state of the radio buttons that allow to select if response of
        only amps, only antennas or amps + antennas should be drawn
    zenith: number
        Value of the slider to select the zenith angle of the incoming signal
        (relevant for antenna response)
    azimuth: number
        Value of the slider to select the azimuth angle of the incoming signal
        (relevant for antenna response)
    station_id: int
        ID of the selected station
    channel_id: int
        ID of the selected channel
    log_checklist: array of strings
        Contains state of the checklist to select logarithmic plotting
        of amplitude and detrending of phase
    """
    if station_id is None or channel_id is None:
        return go.Figure([]), go.Figure([])
    zenith *= units.deg
    azimuth *= units.deg
    if 'log' in log_checklist:
        y_axis_type = 'log'
    else:
        y_axis_type = 'linear'
    frequencies = np.arange(0, 1000, 5) * units.MHz
    response = np.ones((2, frequencies.shape[0]), dtype=complex)
    detector_provider = NuRadioReco.detector.detector_browser.detector_provider.DetectorProvider()
    detector = detector_provider.get_detector()
    if 'amp' in response_type:
        amp_type = detector.get_amplifier_type(station_id, channel_id)
        if amp_type in ['100', '200', '300']:
            amp_response_provider = NuRadioReco.detector.ARIANNA.analog_components.load_amplifier_response(amp_type)
            amp_response = amp_response_provider['gain'](frequencies) * amp_response_provider['phase'](frequencies)
        elif amp_type in ['rno_surface', 'iglu']:
            amp_response_provider = NuRadioReco.detector.RNO_G.analog_components.load_amp_response(amp_type)
            amp_response = amp_response_provider['gain'](frequencies) * amp_response_provider['phase'](frequencies)
        else:
            print('Warning: the specified amplifier was not found')
            amp_response = 1
        response[0] *= amp_response
        response[1] *= amp_response
    if 'antenna' in response_type:
        antenna_model = detector.get_antenna_model(station_id, channel_id, zenith)
        antenna_pattern = antenna_pattern_provider.load_antenna_pattern(antenna_model)
        orientation = detector.get_antenna_orientation(station_id, channel_id)
        VEL = antenna_pattern.get_antenna_response_vectorized(frequencies, zenith, azimuth, *orientation)
        response[0] *= VEL['theta']
        response[1] *= VEL['phi']
        data = [
            go.Scatter(
                x=frequencies / units.MHz,
                y=np.abs(response[0]),
                mode='lines',
                name='Theta component'
            ),
            go.Scatter(
                x=frequencies / units.MHz,
                y=np.abs(response[1]),
                mode='lines',
                name='Phi component'
            )
        ]
        if 'detrend' in log_checklist:
            phase = [scipy.signal.detrend(np.unwrap(np.angle(response[0]))), scipy.signal.detrend(np.unwrap(np.angle(response[1])))]
        else:
            phase = [np.unwrap(np.angle(response[0])), np.unwrap(np.angle(response[1]))]

        phase_data = [
            go.Scatter(
                x=frequencies / units.MHz,
                y=phase[0],
                mode='lines',
                name='Theta component'
            ),
            go.Scatter(
                x=frequencies / units.MHz,
                y=phase[1],
                mode='lines',
                name='Phi component'
            )
        ]
        y_label = 'VEL [m]'
    else:
        data = [go.Scatter(
            x=frequencies / units.MHz,
            y=np.abs(response[1]),
            mode='lines',
            showlegend=False
        )]
        if 'detrend' in log_checklist:
            phase = scipy.signal.detrend(np.unwrap(np.angle(response[1])))
        else:
            phase = np.unwrap(np.angle(response[1]))
        phase_data = [go.Scatter(
            x=frequencies / units.MHz,
            y=phase,
            mode='lines',
            showlegend=False
        )
        ]
        y_label = 'gain'

    fig = go.Figure(data)
    fig.update_layout(
        legend_orientation='h',
        legend=dict(x=.0, y=1.15),
        yaxis_type=y_axis_type,
        xaxis_title='f [MHz]',
        yaxis_title=y_label
    )
    phase_fig = go.Figure(phase_data)
    phase_fig.update_layout(
        legend_orientation='h',
        legend=dict(x=.0, y=1.15),
        xaxis_title='f [MHz]',
        yaxis_title='Phase'
    )
    return fig, phase_fig
