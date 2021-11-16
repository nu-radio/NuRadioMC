import dash
import json
import plotly.subplots
from NuRadioReco.utilities import units, templates, fft
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
"""
Used in the trace from efield and trace frm sim-efield plots. Keep them
commented out for later
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.utilities import geometryUtilities
from NuRadioReco.utilities import trace_utilities
"""
import numpy as np
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from NuRadioReco.eventbrowser.app import app
import os
import NuRadioReco.detector.antennapattern
import NuRadioReco.eventbrowser.dataprovider

provider = NuRadioReco.eventbrowser.dataprovider.DataProvider()

if 'NURADIORECOTEMPLATES' in os.environ:
    template_provider = templates.Templates(os.environ.get('NURADIORECOTEMPLATES'))
else:
    template_provider = templates.Templates('')
antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()

layout = [
    html.Div([
        html.Div([
            dcc.Dropdown(id='dropdown-traces',
                         options=[
                             {'label': 'calibrated trace', 'value': 'trace'},
                             {'label': 'cosmic-ray template', 'value': 'crtemplate'},
                             {'label': 'neutrino template', 'value': 'nutemplate'},
                             {'label': 'envelope', 'value': 'envelope'},
                             {'label': 'from rec. E-field', 'value': 'efield'}
                         ],
                         multi=True,
                         value=["trace"]
                         )
        ], style={'flex': '1'}),
        html.Div([
            dcc.Dropdown(id='dropdown-trace-info',
                         options=[
                             {'label': 'RMS', 'value': 'RMS'},
                             {'label': 'L1', 'value': 'L1'},
                             {'label': 'int. power', 'value':'int_power'},
                             {'label':'frequency RMS', 'value':'freq_RMS'}
                         ],
                         multi=True,
                         value=["RMS", "L1"]
                         )
        ], style={'flex': '1'}),
    ], style={'display': 'flex'}),
    html.Div([
        html.Div([
            html.Div([
                html.Div('Template directory', className='input-group-addon'),
                dcc.Input(id='template-directory-input', placeholder='template directory', className='form-control'),
                html.Div([
                    html.Button('load', id='open-template-button', className='btn btn-default')
                ], className='input-group-btn')
            ], className='input-group', id='template-input-group')
        ], style={'flex': '1'}),
        html.Div('', style={'flex': '1'})
    ], style={'display': 'flex'}),
    dcc.Graph(id='time-traces')
]


@app.callback(
    dash.dependencies.Output('dropdown-traces', 'options'),
    [dash.dependencies.Input('event-counter-slider', 'value'),
     dash.dependencies.Input('filename', 'value'),
     dash.dependencies.Input('station-id-dropdown', 'value')],
    [State('user_id', 'children')]
)
def get_dropdown_traces_options(evt_counter, filename, station_id, juser_id):
    if filename is None or station_id is None:
        return []
    user_id = json.loads(juser_id)
    nurio = provider.get_file_handler(user_id, filename)
    evt = nurio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    options = [
        {'label': 'calibrated trace', 'value': 'trace'},
        {'label': 'cosmic-ray template', 'value': 'crtemplate'},
        {'label': 'neutrino template', 'value': 'nutemplate'},
        {'label': 'envelope', 'value': 'envelope'},
        {'label': 'from rec. E-field', 'value': 'recefield'}
    ]
    if station.get_sim_station() is not None:
        if len(station.get_sim_station().get_electric_fields()) > 0:
            options.append({'label': 'from sim. E-field', 'value': 'simefield'})
    return options


@app.callback(
    Output('template-input-group', 'style'),
    [Input('dropdown-traces', 'value')]
)
def show_template_input(trace_dropdown_options):
    if 'NURADIORECOTEMPLATES' in os.environ:
        return {'display': 'none'}
    if 'crtemplate' in trace_dropdown_options or 'nutemplate' in trace_dropdown_options:
        return {}
    return {'display': 'none'}


def get_L1(a):
    ct = np.array(a[1:]) ** 2
    l1 = np.max(ct) / (np.sum(ct) - np.max(ct))
    return l1


@app.callback(
    dash.dependencies.Output('time-traces', 'figure'),
    [dash.dependencies.Input('event-counter-slider', 'value'),
     dash.dependencies.Input('filename', 'value'),
     dash.dependencies.Input('dropdown-traces', 'value'),
     dash.dependencies.Input('dropdown-trace-info', 'value'),
     dash.dependencies.Input('station-id-dropdown', 'value'),
     dash.dependencies.Input('open-template-button', 'n_clicks_timestamp')],
    [State('user_id', 'children'),
     State('template-directory-input', 'value')])
def update_multi_channel_plot(evt_counter, filename, dropdown_traces, dropdown_info, station_id,
                              open_template_timestamp, juser_id, template_directory):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    nurio = provider.get_file_handler(user_id, filename)
    evt = nurio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    ymax = 0
    n_channels = 0
    plot_titles = []
    trace_start_times = []
    n_rows = station.get_number_of_channels()
    fig = plotly.subplots.make_subplots(rows=n_rows, cols=2,
                                        shared_xaxes=True, shared_yaxes=False,
                                        vertical_spacing=0.05/n_rows, subplot_titles=plot_titles)
    for i, channel in enumerate(station.iter_channels()):
        n_channels += 1
        trace = channel.get_trace() / units.mV
        ymax = max(ymax, np.max(np.abs(trace)))
        plot_titles.append('Channel {}'.format(channel.get_id()))
        plot_titles.append('Channel {}'.format(channel.get_id()))
        trace_start_times.append(channel.get_trace_start_time())
        if channel.get_trace() is not None:
            trace = channel.get_trace() / units.mV
            ymax = max(ymax, np.max(np.abs(trace)))
    if np.min(trace_start_times) > 1000. * units.ns:
        trace_start_time_offset = np.floor(np.min(trace_start_times) / 1000.) * 1000.
    else:
        trace_start_time_offset = 0
    channel_ids = []
    for i, channel in enumerate(station.iter_channels()):
        channel_ids.append(channel.get_id())
    channel_ids = sorted(channel_ids)
    if 'trace' in dropdown_traces:
        for i, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            tt = channel.get_times() - trace_start_time_offset / units.ns
            if channel.get_trace() is None:
                continue
            trace = channel.get_trace() / units.mV
            fig.append_trace(plotly.graph_objs.Scatter(
                x=tt,
                y=trace,
                opacity=0.7,
                marker={
                    'color': colors[i % len(colors)],
                    'line': {'color': colors[i % len(colors)]}
                },
                name=str(channel_id)
            ), i + 1, 1)
            if 'RMS' in dropdown_info:
                fig.add_annotation(
                    text=r'mu = {:.2g}, STD={:.2g}'.format(np.mean(trace), np.std(trace)),
                    x=0.99, y=0.98, xanchor='right', yanchor='top', showarrow=False,
                    xref='x domain', yref='y domain',
                    row=i+1, col=1)
            if 'int_power' in dropdown_info:
                fig.add_annotation(
                    text=r'E ~ {:.3g}'.format(np.sum(trace**2)),
                    x=0.99, y=0.3, xanchor='right', yanchor='top', showarrow=False,
                    xref='x domain', yref='y domain',
                    row=i+1, col=1)
    if 'envelope' in dropdown_traces:
        for i, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            if channel.get_trace() is None:
                continue
            trace = channel.get_trace() / units.mV
            from scipy import signal
            yy = np.abs(signal.hilbert(trace))
            fig.append_trace(plotly.graph_objs.Scatter(
                x=channel.get_times() - trace_start_time_offset / units.ns,
                y=yy,
                opacity=0.7,
                line=dict(
                    width=4,
                    dash='dot'),  
                marker={
                    'color': colors[i % len(colors)],
                    'line': {'color': colors[i % len(colors)]}
                },
                name=i
            ), i + 1, 1)
    if 'crtemplate' in dropdown_traces:
        if 'NURADIORECOTEMPLATES' not in os.environ:  # if the environment variable is not set, we have to ask the user to specify the template location
            template_provider.set_template_directory(template_directory)
        ref_template = template_provider.get_cr_ref_template(station_id)
        if (station.has_parameter(stnp.cr_xcorrelations)):
            ref_templates = template_provider.get_set_of_cr_templates(station_id,
                                                                      n=station.get_parameter(stnp.cr_xcorrelations)[
                                                                          'number_of_templates'])
        for i, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            if (channel.has_parameter(chp.cr_xcorrelations)):
                key = channel.get_parameter(chp.cr_xcorrelations)['cr_ref_xcorr_template']
                ref_template = ref_templates[key][channel.get_id()]
            times = channel.get_times() - trace_start_time_offset
            trace = channel.get_trace()
            if trace is None:
                continue
            xcorr = channel.get_parameter(chp.cr_xcorrelations)['cr_ref_xcorr']
            xcorrpos = channel.get_parameter(chp.cr_xcorrelations)['cr_ref_xcorr_time']
            dt = times[1] - times[0]
            xcorr_max = xcorr
            if (channel.has_parameter(chp.cr_xcorrelations)):
                xcorr_max = channel.get_parameter(chp.cr_xcorrelations)['cr_ref_xcorr_max']
            flip = np.sign(xcorr_max)
            tttemp = np.arange(0, len(ref_template) * dt, dt)
            yy = flip * ref_template * np.abs(trace).max()
            fig.append_trace(plotly.graph_objs.Scatter(
                x=tttemp[:len(trace)] / units.ns,
                y=yy[:len(trace)] / units.mV,
                opacity=0.7,
                line=dict(
                    width=2,
                    dash='dot'),  # dash options include 'dash', 'dot', and 'dashdot'
                marker={
                    'color': colors[i % len(colors)],
                    'line': {'color': colors[i % len(colors)]}
                },
                name=i
            ), i + 1, 1)
            template_spectrum = fft.time2freq(yy, channel.get_sampling_rate())
            template_freqs = np.fft.rfftfreq(len(yy), dt)
            template_freq_mask = (template_freqs > channel.get_frequencies()[0]) & (template_freqs < (channel.get_frequencies()[-1]))
            fig.append_trace(plotly.graph_objs.Scatter(
                x=template_freqs[template_freq_mask] / units.MHz,
                y=np.abs(template_spectrum)[template_freq_mask] / units.mV,
                line=dict(
                    width=2,
                    dash='dot'
                ),
                name=i
            ), i + 1, 2)
            fig.append_trace(
                plotly.graph_objs.Scatter(
                    x=[0.9 * times.max() / units.ns],
                    y=[0.8 * trace.max() / units.mV],
                    mode='text',
                    text=['cr xcorr= {:.2f}, {:.02f}'.format(xcorr, xcorr_max)],
                    textposition='top center'
                ),
                i + 1, 1)
            fig.append_trace(
                plotly.graph_objs.Scatter(
                    x=[0.9 * times.max() / units.ns],
                    y=[0.5 * trace.max() / units.mV],
                    mode='text',
                    text=['t = {:.0f}ns'.format(xcorrpos)],
                    textposition='top center'
                ),
                i + 1, 1)
    if 'nutemplate' in dropdown_traces:
        if 'NURADIORECOTEMPLATES' not in os.environ:  # if the environment variable is not set, we have to ask the user to specify the template location
            template_provider.set_template_directory(template_directory)
        ref_template = template_provider.get_nu_ref_template(station_id)
        for i, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            times = channel.get_times()
            trace = channel.get_trace()
            if trace is None:
                continue
            xcorr = channel.get_parameter(chp.nu_xcorrelations)['nu_ref_xcorr']
            xcorrpos = channel.get_parameter(chp.nu_xcorrelations)['nu_ref_xcorr_time']
            dt = times[1] - times[0]
            flip = np.sign(xcorr)
            tttemp = np.arange(0, len(ref_template) * dt, dt)
            yy = flip * ref_template * np.abs(trace).max()
            fig.append_trace(plotly.graph_objs.Scatter(
                x=tttemp[:len(trace)] / units.ns,
                y=yy[:len(trace)] / units.mV,
                opacity=0.7,
                line=dict(
                    width=4,
                    dash='dot'),  # dash options include 'dash', 'dot', and 'dashdot'
                marker={
                    'color': colors[i % len(colors)],
                    'line': {'color': colors[i % len(colors)]}
                },
                name=i
            ), i + 1, 1)
            template_spectrum = fft.time2freq(yy, channel.get_sampling_rate())
            template_freqs = np.fft.rfftfreq(len(yy), dt)
            template_freq_mask = (template_freqs > channel.get_frequencies()[0]) & (template_freqs < (channel.get_frequencies()[-1]))
            fig.append_trace(plotly.graph_objs.Scatter(
                x=template_freqs[template_freq_mask] / units.MHz,
                y=np.abs(template_spectrum)[template_freq_mask] / units.mV,
                line=dict(
                    width=2,
                    dash='dot'
                ),
                name=i
            ), i + 1, 2)

            fig.append_trace(
                plotly.graph_objs.Scatter(
                    x=[0.9 * times.max() / units.ns],
                    y=[0.8 * trace.max() / units.mV],
                    mode='text',
                    text=['nu xcorr= {:.2f}'.format(xcorr)],
                    textposition='top center'
                ),
                i + 1, 1)
            fig.append_trace(
                plotly.graph_objs.Scatter(
                    x=[0.9 * times.max() / units.ns],
                    y=[0.5 * trace.max() / units.mV],
                    mode='text',
                    text=['t = {:.0f}ns'.format(xcorrpos)],
                    textposition='top center'
                ),
                i + 1, 1)
    """
    if 'recefield' in dropdown_traces or 'simefield' in dropdown_traces:
        det.update(station.get_station_time())
    if 'recefield' in dropdown_traces:
        channel_ids = []
        for channel in station.iter_channels():
            channel_ids.append(channel.get_id())
        for electric_field in station.get_electric_fields():
            for i_trace, trace in enumerate(
                    trace_utilities.get_channel_voltage_from_efield(station, electric_field, channel_ids, det,
                                                                    station.get_parameter(stnp.zenith),
                                                                    station.get_parameter(stnp.azimuth),
                                                                    antenna_pattern_provider)):
                direction_time_delay = geometryUtilities.get_time_delay_from_direction(
                    station.get_parameter(stnp.zenith), station.get_parameter(stnp.azimuth),
                    det.get_relative_position(station.get_id(), channel_ids[i_trace]) - electric_field.get_position())
                time_shift = direction_time_delay - trace_start_time_offset
                plotly.graph_objs.append_trace(plotly.graph_objs.Scatter(
                    x=(electric_field.get_times() + time_shift) / units.ns,
                    y=fft.freq2time(trace, electric_field.get_sampling_rate()) / units.mV,
                    line=dict(
                        dash='solid',
                        color=colors[i_trace % len(colors)]
                    ),
                    opacity=.5
                ), i_trace + 1, 1)
                plotly.graph_objs.append_trace(plotly.graph_objs.Scatter(
                    x=electric_field.get_frequencies() / units.MHz,
                    y=np.abs(trace) / units.mV,
                    line=dict(
                        dash='solid',
                        color=colors[i_trace % len(colors)]
                    ),
                    opacity=.5
                ), i_trace + 1, 2)
    if 'simefield' in dropdown_traces:
        sim_station = station.get_sim_station()
        for i_channel, channel in enumerate(station.iter_channels()):
            for electric_field in sim_station.get_electric_fields_for_channels([channel.get_id()]):
                trace = \
                    trace_utilities.get_channel_voltage_from_efield(sim_station, electric_field, [channel.get_id()],
                                                                    det,
                                                                    electric_field.get_parameter(efp.zenith),
                                                                    electric_field.get_parameter(efp.azimuth),
                                                                    antenna_pattern_provider)[0]
                channel = station.get_channel(channel.get_id())
                if station.is_cosmic_ray():
                    direction_time_delay = geometryUtilities.get_time_delay_from_direction(
                        sim_station.get_parameter(stnp.zenith), sim_station.get_parameter(stnp.azimuth),
                        det.get_relative_position(sim_station.get_id(),
                                                  channel.get_id()) - electric_field.get_position())
                    time_shift = direction_time_delay - trace_start_time_offset
                else:
                    time_shift = - trace_start_time_offset
                fig.append_trace(plotly.graph_objs.Scatter(
                    x=(electric_field.get_times() + time_shift) / units.ns,
                    y=fft.freq2time(trace, electric_field.get_sampling_rate()) / units.mV,
                    line=dict(
                        dash='solid',
                        color=colors[i_channel % len(colors)]
                    ),
                    opacity=.5
                ), i_channel + 1, 1)
                fig.append_trace(plotly.graph_objs.Scatter(
                    x=electric_field.get_frequencies() / units.MHz,
                    y=np.abs(trace) / units.mV,
                    line=dict(
                        dash='solid',
                        color=colors[i_channel % len(colors)]
                    ),
                    opacity=.5
                ), i_channel + 1, 2)
    """
    for i, channel_id in enumerate(channel_ids):
        channel = station.get_channel(channel_id)
        fig['layout']['yaxis{:d}'.format(i * 2 + 1)].update(range=[-ymax, ymax])
        fig['layout']['yaxis{:d}'.format(i * 2 + 1)].update(
            title='<b>Ch. {}</b><br>voltage [mV]'.format(channel_id)
        )

        if channel.get_trace() is None:
            continue
        spec = channel.get_frequency_spectrum()
        ff = channel.get_frequencies()
        fig.append_trace(plotly.graph_objs.Scatter(
            x=ff / units.MHz,
            y=np.abs(spec) / units.mV,
            opacity=0.7,
            marker={
                'color': colors[i % len(colors)],
                'line': {'color': colors[i % len(colors)]}
            },
            name=i
        ), i + 1, 2)
        if 'L1' in dropdown_info:
            fig.append_trace(
                plotly.graph_objs.Scatter(
                    x=[0.9 * ff.max() / units.MHz],
                    y=[0.8 * np.abs(spec).max() / units.mV],
                    mode='text',
                    text=['max L1 = {:.2f}'.format(get_L1(spec))],
                    textposition='top center'
                ),
                i + 1, 2)
        if 'freq_RMS' in dropdown_info:
            fig.add_hline(
                .25 * np.max(np.abs(spec)[5:]) / units.MHz,
                row=i+1, col=2
            )
    if trace_start_time_offset > 0:
        fig['layout']['xaxis1'].update(title='time [ns] - {:.0f} ns'.format(trace_start_time_offset))
    else:
        fig['layout']['xaxis1'].update(title='time [ns]')
    fig['layout']['xaxis1'].update(side='top', showticklabels=True)
    fig['layout']['xaxis2'].update(side='top', showticklabels=True)
    fig['layout']['xaxis2'].update(title='frequency [MHz]')
    for i in range(2):
        last_xaxis = 'xaxis{}'.format(station.get_number_of_channels() * 2 + i - 1)
        first_xaxis = 'xaxis{}'.format(i+1)
        fig['layout'][last_xaxis].update(title=fig['layout'][first_xaxis]['title'])
    fig['layout'].update(height=n_channels * 150)
    fig['layout'].update(showlegend=False)
    return fig
