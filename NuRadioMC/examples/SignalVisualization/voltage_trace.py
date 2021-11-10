from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import dash
import plotly.subplots
import plotly.graph_objs as go
import numpy as np
import json
from NuRadioReco.utilities import units, fft
import NuRadioReco.detector.antennapattern
import NuRadioReco.detector.ARIANNA.analog_components
import NuRadioReco.detector.RNO_G.analog_components
import radiotools.helper as hp
from app import app
import antennas
import amplifiers
import scipy.signal
import io

DEFAULT_CHART_CONFIG = {
    'modeBarButtons': [
        [
            'toImage',
            'sendDataToCloud',
            'zoom2d',
            'pan2d',
            'zoomIn2d',
            'zoomOut2d',
            'autoScale2d',
            'resetScale2d',
            'toggleSpikelines',
            'hoverClosestCartesian',
            'hoverCompareCartesian'
        ]
    ]
}

antennapattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()

antenna_panel = html.Div([
    html.Div([
        html.Div('Antenna', className='panel-heading'),
        html.Div([
            html.Div([
                html.Div('Antenna Type'),
                html.Div([
                    html.Div([
                        dcc.Dropdown(
                            id='antenna-type-radio-items',
                            options=antennas.antenna_options,
                            value='bicone_v8_InfFirn',
                            multi=False
                        )
                    ], style={'flex': '1'}),
                    html.Div([
                        html.Div([
                            html.Div('', className='icon-question-circle-o')
                        ], className='popup-symbol'),
                        html.Div('', id='antenna-type-popup', className='popup-box')
                    ], className='popup-container', style={'flex': 'none'})
                ], style={'display': 'flex'})
            ], className='input-group'),
            html.Div([
                html.Div('Signal Zenith Angle'),
                dcc.Slider(
                    id='signal-zenith-slider',
                    min=0,
                    max=180,
                    step=5,
                    value=45,
                    marks={
                        0: '0°',
                        45: '45°',
                        90: '90°',
                        135: '135°',
                        180: '180°'
                    }
                )
            ], className='input-group'),
            html.Div([
                html.Div('Signal Azimuth'),
                dcc.Slider(
                    id='signal-azimuth-slider',
                    min=0,
                    max=360,
                    step=10,
                    value=180,
                    marks={
                        0: '0°',
                        90: '90°',
                        180: '180°',
                        270: '270°',
                        360: '360°'
                    }
                )
            ], className='input-group')
        ], className='panel-body')
    ], className='panel panel-default')
])

amplifier_panel = html.Div([
    html.Div([
        html.Div('Amplifier', className='panel-heading'),
        html.Div([
            html.Div([
                html.Div('Amplifier'),
                html.Div([
                    html.Div([
                        dcc.Dropdown(
                            id='amplifier-type-dropdown',
                            options=amplifiers.amplifier_options,
                            value=None,
                            multi=False
                        )
                    ], style={'flex': 1}),
                    html.Div([
                        html.Div([
                            html.Div('', className='icon-question-circle-o')
                        ], className='popup-symbol'),
                        html.Div('', id='amplifier-popup', className='popup-box')
                    ], className='popup-container', style={'flex': 'none'})
                ], style={'display': 'flex'})
            ], className='input-group'),
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id='filter-toggle-checklist',
                            options=[
                                {'label': 'Filter', 'value': 'filter'}
                            ],
                            value=[]
                        )
                    ], style={'flex': '1'}),
                    html.Div([
                        html.Div([
                            html.Div('', className='icon-question-circle-o')
                        ], className='popup-symbol'),
                        html.Div(('Apply a 10th order Butterworth filter '
                            'to the channel spectrum. Phase is not affected'
                        ), className='popup-box')
                    ], className='popup-container', style={'flex': 'none'})
                ], style={'display': 'flex'}),
                dcc.RangeSlider(
                    id='filter-band-range-slider',
                    min=0,
                    max=1.,
                    step=.05,
                    value=[0, 1.],
                    marks={
                        0: '0MHz',
                        .2: '200MHz',
                        .4: '400MHz',
                        .6: '600MHz',
                        .8: '800MHz',
                        1.: '1GHz'
                    }
                )
            ], className='input-group')
        ], className='panel-body')
    ], className='panel panel-default')
])

voltage_plot_panel = html.Div([
    html.Div([
        html.Div('Voltage', className='panel-heading'),
        html.A(
            'Download Data',
            id='download-link',
            download="voltage_trace.csv",
            href="",
            target="_blank"
            ),
        html.Div([
            dcc.Graph(id='voltage-plots')
        ], className='panel-body')
    ], className='panel panel-default')
])

signal_direction_panel = html.Div([
    html.Div([
        html.Div('Signal Direction', className='panel-heading'),
        html.Div([
        dcc.Graph(id='signal-direction-plot')
        ], className='panel-body')
    ], className='panel panel-default')
])
hardware_response_panel = html.Div([
    html.Div([
        html.Div('Detector Response', className='panel-heading'),
        html.Div([
            dcc.Graph(id='detector-response-plot')
        ], className='panel-body')
    ], className='panel panel-default')
])


@app.callback(
    [Output('voltage-plots', 'figure'),
    Output('detector-response-plot', 'figure'),
    Output('download-link', 'href')],
    [Input('efield-trace-storage', 'children'),
    Input('antenna-type-radio-items', 'value'),
    Input('signal-zenith-slider', 'value'),
    Input('signal-azimuth-slider', 'value'),
    Input('amplifier-type-dropdown', 'value'),
    Input('filter-toggle-checklist', 'value'),
    Input('filter-band-range-slider', 'value')],
    [State('sampling-rate-slider', 'value')]
)
def update_voltage_plot(
    electric_field,
    antenna_type,
    signal_zenith,
    signal_azimuth,
    amplifier_type,
    filter_toggle,
    filter_band,
    sampling_rate
):
    samples = int(512 * sampling_rate)
    electric_field = json.loads(electric_field)
    if electric_field is None:
        return {}, {}, ""
    antenna_pattern = antennapattern_provider.load_antenna_pattern(antenna_type)
    freqs = np.fft.rfftfreq(samples, 1. / sampling_rate)
    times = np.arange(samples) / sampling_rate
    signal_zenith = signal_zenith * units.deg
    signal_azimuth = signal_azimuth * units.deg

    antenna_response = antenna_pattern.get_antenna_response_vectorized(
        freqs,
        signal_zenith,
        signal_azimuth,
        0.,
        0.,
        90.*units.deg,
        180.*units.deg
    )
    detector_response_theta = antenna_response['theta']
    detector_response_phi = antenna_response['phi']
    channel_spectrum = antenna_response['theta'] * fft.time2freq(electric_field['theta'], sampling_rate) + antenna_response['phi'] * fft.time2freq(electric_field['phi'], sampling_rate)
    if amplifier_type is not None:
        if amplifier_type == 'iglu' or amplifier_type == 'rno_surface':
            amp_response = NuRadioReco.detector.RNO_G.analog_components.load_amp_response(amplifier_type)
            amplifier_response = amp_response['gain'](freqs) * amp_response['phase'](freqs)
        else:
            amp_response = NuRadioReco.detector.ARIANNA.analog_components.load_amplifier_response(amplifier_type)
            amplifier_response = amp_response['gain'](freqs) * amp_response['phase'](freqs)
        channel_spectrum = channel_spectrum * amplifier_response
        detector_response_theta *= amplifier_response
        detector_response_phi *= amplifier_response
    if 'filter' in filter_toggle:
        mask = freqs > 0
        b, a = scipy.signal.butter(10, filter_band, 'bandpass', analog=True)
        w, h = scipy.signal.freqs(b, a, freqs[mask])
        channel_spectrum[mask] = channel_spectrum[mask] * np.abs(h)
        detector_response_theta[mask] *= np.abs(h)
        detector_response_phi[mask] *= np.abs(h)
    channel_trace = fft.freq2time(channel_spectrum, sampling_rate)
    max_time = times[np.argmax(np.abs(channel_trace))]
    fig = plotly.subplots.make_subplots(rows=1, cols=2,
        shared_xaxes=False, shared_yaxes=False,
        vertical_spacing=0.01, subplot_titles=['Time Trace', 'Spectrum'])
    fig.append_trace(go.Scatter(
        x=times / units.ns,
        y=channel_trace / units.mV,
        name='U (t)'
    ), 1, 1)
    fig.append_trace(go.Scatter(
        x=freqs / units.MHz,
        y=np.abs(channel_spectrum) / (units.mV / units.GHz),
        name='U (f)'
    ), 1, 2)
    fig.update_xaxes(title_text='t [ns]', range=[max_time - 50.*units.ns, max_time + 50.*units.ns], row=1, col=1)
    fig.update_xaxes(title_text='f [MHz]', row=1, col=2)
    fig.update_yaxes(title_text='U [mV]', row=1, col=1)
    fig.update_yaxes(title_text='U [mV/GHz]', row=1, col=2)

    fig2 = plotly.subplots.make_subplots(rows=1, cols=2,
        shared_xaxes=False, shared_yaxes=True,
        vertical_spacing=0.01, subplot_titles=['Theta', 'Phi'])
    fig2.append_trace(go.Scatter(
        x=freqs / units.MHz,
        y=np.abs(detector_response_theta),
        name='Theta'
    ), 1, 1)
    fig2.append_trace(go.Scatter(
        x=freqs / units.MHz,
        y=np.abs(detector_response_phi),
        name='Phi'
    ), 1, 2)
    fig2.update_xaxes(title_text='f [MHz]', row=1, col=1)
    fig2.update_xaxes(title_text='f [MHz]', row=1, col=2)
    fig2.update_yaxes(title_text='VEL [m]', row=1, col=1)
    fig2.update_yaxes(title_text='VEL [m]', row=1, col=2)

    s = io.StringIO()
    np.savetxt(s, np.array([times, channel_trace]).T, delimiter=',')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + s.getvalue()

    return [fig, fig2, csv_string]


@app.callback(
    Output('signal-direction-plot', 'figure'),
    [Input('signal-zenith-slider', 'value'),
    Input('signal-azimuth-slider', 'value'),
    Input('antenna-type-radio-items', 'value')]
)
def update_signal_direction_plot(zenith, azimuth, antenna_type):
    zenith = zenith * units.deg
    azimuth = azimuth * units.deg
    signal_direction = hp.spherical_to_cartesian(zenith, azimuth)
    data = []
    data.append(go.Scatter3d(
        x=[0, signal_direction[0]],
        y=[0, signal_direction[1]],
        z=[0, signal_direction[2]],
        mode='lines',
        name='Signal Direction'
    ))
    data.append(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, 1],
        mode='lines',
        name='Antenna Orientation'
    ))
    rot = hp.spherical_to_cartesian(90.*units.deg, 180*units.deg)
    data.append(go.Scatter3d(
        x=[0, rot[0]],
        y=[0, rot[1]],
        z=[0, rot[2]],
        mode='lines',
        name='Antenna Rotation'
    ))
    if antenna_type == 'createLPDA_100MHz_InfFirn':
        data.append(go.Mesh3d(
            x=[0, 0, 0],
            y=[-.25, 0, .25],
            z=[0, .75, 0],
            delaunayaxis='x',
            opacity=.5,
            color='black'
            ))
    else:
        d_angle = 30
        angles = np.arange(0, 360, d_angle) * units.deg
        r = .05
        cylinder_points = []
        i = []
        j = []
        k = []
        for i_angle, angle in enumerate(angles):
            cylinder_points.append([r * np.cos(angle), r * np.sin(angle), -.5])
            cylinder_points.append([r * np.cos(angle), r * np.sin(angle), .5])
            cylinder_points.append([r * np.cos(angle + d_angle * units.deg), r * np.sin(angle + d_angle * units.deg), -.5])
            cylinder_points.append([r * np.cos(angle), r * np.sin(angle), .5])
            cylinder_points.append([r * np.cos(angle + d_angle * units.deg), r * np.sin(angle + d_angle * units.deg), -.5])
            cylinder_points.append([r * np.cos(angle + d_angle * units.deg), r * np.sin(angle + d_angle * units.deg), .5])
            i.append(6 * i_angle)
            j.append(6 * i_angle + 1)
            k.append(6 * i_angle + 2)
            i.append(6 * i_angle + 3)
            j.append(6 * i_angle + 4)
            k.append(6 * i_angle + 5)
        cylinder_points = np.array(cylinder_points)
        data.append(go.Mesh3d(
            x=cylinder_points[:, 0],
            y=cylinder_points[:, 1],
            z=cylinder_points[:, 2],
            i=i,
            j=j,
            k=k,
            color='black',
            opacity=.5
        ))

    fig = go.Figure(
        data=data
    )
    fig.update_layout(
        scene=dict(
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        zaxis=dict(range=[-1, 1]),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1)
        ),
        showlegend=True,
        legend=dict(x=-.1, y=1.1),
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        )
    )
    return fig


@app.callback(
    Output('antenna-type-popup', 'children'),
    [Input('antenna-type-radio-items', 'value')]
)
def get_antenna_type_tooltip(selected_antenna):
    for antenna in antennas.antenna_options:
        if antenna['value'] == selected_antenna:
            return antenna['description']
    return ''


@app.callback(
    Output('amplifier-popup', 'children'),
    [Input('amplifier-type-dropdown', 'value')]
)
def get_amplifier_tooltip(selected_amplifier):
    for amp in amplifiers.amplifier_options:
        if amp['value'] == selected_amplifier:
            return amp['description']
    return ''
