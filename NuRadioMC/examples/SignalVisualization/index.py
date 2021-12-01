from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import dash
import json
import plotly.subplots
import plotly.graph_objs as go
import numpy as np
from NuRadioReco.utilities import units, fft
import NuRadioMC.utilities.attenuation
import NuRadioMC.SignalGen.askaryan
import voltage_trace
from app import app

app.title = 'Radio Signal Simulator'

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Div('Neutrino Settings', className='panel-heading'),
                html.Div([
                    html.Div([
                        html.Div('Energy'),
                        dcc.Slider(
                            id='energy-slider',
                            min=16,
                            max=20,
                            step=.25,
                            value=18,
                            marks={
                                16: '10PeV',
                                17: '100PeV',
                                18: '1EeV',
                                19: '10EeV',
                                20: '100EeV'
                            }
                        )
                    ], className='input-group'),
                    html.Div([
                        html.Div('Viewing angle'),
                        html.Div([
                            html.Div([
                                dcc.Slider(
                                    id='viewing-angle-slider',
                                    min=-10,
                                    max=10,
                                    step=1,
                                    value=2,
                                    marks={
                                        -10: '-10°',
                                        -5: '-5°',
                                        -2: '-2°',
                                        0: '0°',
                                        2: '2°',
                                        5: '5°',
                                        10: '10°'
                                    }
                                )
                            ], style={'flex': '1'}),
                            html.Div([
                                html.Div([
                                    html.Div('', className='icon-question-circle-o')
                                ], className='popup-symbol'),
                                html.Div(('Angle between the emission direction '
                                'of the signal and the Cherenkov angle'),
                                className='popup-box')
                            ], className='popup-container', style={'flex': 'none'})
                        ], style={'display': 'flex'})
                    ], className='input-group'),
                    html.Div([
                        html.Div('Polarization Angle'),
                        html.Div([
                            html.Div([
                            dcc.Slider(
                                id='polarization-angle-slider',
                                min=-180,
                                max=180,
                                step=5,
                                value=0,
                                marks={
                                    -180: '-180°',
                                    -90: '-90°',
                                    -45: '-45°',
                                    0: '0°',
                                    45: '45°',
                                    90: '90°',
                                    180: '180°'
                                }
                            )
                        ], style={'flex': '1'}),
                        html.Div([
                            html.Div([
                                html.Div('', className='icon-question-circle-o')
                            ], className='popup-symbol'),
                            html.Div(('Angle between the electric field vector and'
                                ' e_theta'), className='popup-box')
                        ], className='popup-container', style={'flex': 'none'})
                    ], style={'display': 'flex'})
                    ], className='input-group'),
                    html.Div([
                        html.Div('Shower Type'),
                        dcc.RadioItems(
                            id='shower-type-radio-items',
                            options=[
                                {'label': 'Hadronic', 'value': 'HAD'},
                                {'label': 'Electro-Magnetic', 'value': 'EM'}
                            ],
                            value='HAD',
                            labelStyle={'padding':'0 5px'}
                        )
                    ], className='input-group'),
                ], className='panel-body')
            ], className='panel panel-default'),
            html.Div([
                html.Div('Simulation Settings', className='panel-heading'),
                html.Div([
                    html.Div([
                        html.Div('Shower Model'),
                        dcc.Dropdown(
                            id='shower-model-dropdown',
                            options=[
                                {'label': 'ARZ2020', 'value': 'ARZ2020'},
                                {'label': 'ARZ2019', 'value': 'ARZ2019'},
                                {'label': 'Alvarez2009', 'value': 'Alvarez2009'},
                                {'label': 'Alvarez2000', 'value': 'Alvarez2000'},
                                {'label': 'ZHS1992', 'value': 'ZHS1992'}
                            ],
                            multi=False,
                            value='ARZ2020'
                        )
                    ], className='input-group'),
                    html.Div([
                        html.Div('Sampling Rate'),
                        dcc.Slider(
                            id='sampling-rate-slider',
                            min=1.,
                            max=5.,
                            step=.5,
                            value=2.,
                            marks={
                                1: '1GHz',
                                2: '2GHz',
                                3: '3GHz',
                                4: '4GHz',
                                5: '5GHz'
                            }
                        )
                    ], className='input-group')
                ], className='panel-body')
            ], className='panel panel-default'),
            html.Div([
                html.Div('Propagation', className='panel-heading'),
                html.Div([
                    html.Div([
                        html.Div('Propagation Length'),
                        dcc.Slider(
                            id='propagation-length-slider',
                            min=0,
                            max=5,
                            step=.1,
                            value=0,
                            marks={
                                0: '0',
                                1: '1km',
                                2: '2km',
                                3: '3km',
                                4: '4km',
                                5: '5km'
                            }
                        )
                    ], className='input-group'),
                    html.Div([
                        html.Div('Attenuation Model'),
                        dcc.RadioItems(
                            id='attenuation-model-radio-items',
                            options=[
                                {'label': 'Greenland', 'value': 'GL1'},
                                {'label': 'South Pole', 'value': 'SP1'},
                                {'label': 'Moore´s Bay', 'value': 'MB1'}
                            ],
                            value='GL1',
                            labelStyle={'padding': '0 5px'}
                        )
                    ], className='input-group')
                ], className='panel-body')
            ], className='panel panel-default'),
            voltage_trace.antenna_panel,
            voltage_trace.amplifier_panel,
            voltage_trace.signal_direction_panel
        ], style={'flex':'1'}),
        html.Div([
            html.Div([
                html.Div('Electric Field', className='panel-heading'),
                html.Div([
                    dcc.Graph(id='electric-field-plot')
                ], className='panel-body')
            ], className='panel panel-default'),
            voltage_trace.voltage_plot_panel,
            voltage_trace.hardware_response_panel
            ], style={'flex':'4'})
    ], style={'display': 'flex'}),
    html.Div(id='efield-trace-storage', children=json.dumps(None), style={'display': 'none'})
])


@app.callback(
    [Output('electric-field-plot', 'figure'),
    Output('efield-trace-storage', 'children')],
    [Input('energy-slider', 'value'),
    Input('viewing-angle-slider', 'value'),
    Input('shower-type-radio-items', 'value'),
    Input('polarization-angle-slider', 'value'),
    Input('shower-model-dropdown', 'value'),
    Input('propagation-length-slider', 'value'),
    Input('attenuation-model-radio-items', 'value'),
    Input('sampling-rate-slider', 'value')]
)
def update_electric_field_plot(
    log_energy,
    viewing_angle,
    shower_type,
    polarization_angle,
    model,
    propagation_length,
    attenuation_model,
    sampling_rate):

    viewing_angle = viewing_angle * units.deg
    polarization_angle = polarization_angle * units.deg
    propagation_length  = propagation_length * units.km
    energy = np.power(10., log_energy)
    samples = int(512 * sampling_rate)
    ior = 1.78
    cherenkov_angle = np.arccos(1./ior)
    distance = 1.*units.km
    try:
        efield_spectrum = NuRadioMC.SignalGen.askaryan.get_frequency_spectrum(
            energy,
            cherenkov_angle + viewing_angle,
            samples,
            1./sampling_rate,
            shower_type,
            ior,
            distance,
            model,
            same_shower=True
        )
    except:
        efield_spectrum = NuRadioMC.SignalGen.askaryan.get_frequency_spectrum(
            energy,
            cherenkov_angle + viewing_angle,
            samples,
            1./sampling_rate,
            shower_type,
            ior,
            distance,
            model,
            same_shower=False
        )
    freqs = np.fft.rfftfreq(samples, 1./sampling_rate)
    if propagation_length > 0:
        attenuation_length = NuRadioMC.utilities.attenuation.get_attenuation_length(200., freqs, attenuation_model)
        efield_spectrum *= np.exp(-propagation_length/attenuation_length)
    efield_spectrum_theta = efield_spectrum * np.cos(polarization_angle)
    efield_spectrum_phi = efield_spectrum * np.sin(polarization_angle)
    efield_trace = fft.freq2time(efield_spectrum, sampling_rate)
    efield_trace_theta = efield_trace * np.cos(polarization_angle)
    efield_trace_phi = efield_trace * np.sin(polarization_angle)
    times = np.arange(samples) / sampling_rate

    fig = plotly.subplots.make_subplots(rows=1, cols=2,
        shared_xaxes=False, shared_yaxes=False,
        vertical_spacing=0.01, subplot_titles=['Time Trace', 'Spectrum'])
    fig.append_trace(go.Scatter(
        x=times/units.ns,
        y=efield_trace_theta/(units.mV/units.m),
        name='E_theta (t)'
    ),1,1)
    fig.append_trace(go.Scatter(
        x=times/units.ns,
        y=efield_trace_phi/(units.mV/units.m),
        name='E_phi (t)'
    ),1,1)
    fig.append_trace(go.Scatter(
        x=freqs/units.MHz,
        y=np.abs(efield_spectrum_theta)/(units.mV/units.m/units.GHz),
        name='E_theta (f)'
    ),1,2)
    fig.append_trace(go.Scatter(
        x=freqs/units.MHz,
        y=np.abs(efield_spectrum_phi)/(units.mV/units.m/units.GHz),
        name='E_phi (f)'
    ),1,2)
    max_time = times[np.argmax(np.sqrt(efield_trace_phi**2+efield_trace_theta**2))]
    fig.update_xaxes(title_text='t [ns]', range=[max_time-50*units.ns, max_time+50*units.ns], row=1, col=1)
    fig.update_xaxes(title_text='f [MHz]', row=1, col=2)
    fig.update_yaxes(title_text='E[mV/m]', row=1, col=1)
    fig.update_yaxes(title_text='E [mV/m/GHz]', row=1, col=2)
    return [fig, json.dumps({'theta':efield_trace_theta.tolist(), 'phi':efield_trace_phi.tolist()})]

app.run_server(debug=False, port=8080)
