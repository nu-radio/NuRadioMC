from __future__ import absolute_import, division, print_function  # , unicode_literals
from dash.dependencies import Input, Output, State
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import plotly
from plotly import tools
import json
from app import app
import dataprovider
from NuRadioReco.utilities import units
from NuRadioReco.utilities import templates
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
import numpy as np
import logging
logger = logging.getLogger('traces')

provider = dataprovider.DataProvider()
template_provider = templates.Templates()

layout = html.Div([
    html.Div(id='trigger-trace', style={'display': 'none'}),
    dcc.Dropdown(id='dropdown-traces',
        options=[
            {'label': 'calibrated trace', 'value': 'trace'},
            {'label': 'cosmic-ray template', 'value': 'crtemplate'},
            {'label': 'neutrino template', 'value': 'nutemplate'},
            {'label': 'envelope', 'value': 'envelope'}
        ],
        multi=True,
        value=["trace"]
    ),
    dcc.Dropdown(id='dropdown-trace-info',
        options=[
            {'label': 'RMS', 'value': 'RMS'},
            {'label': 'L1', 'value': 'L1'}
        ],
        multi=True,
        value=["RMS", "L1"]
    ),
    html.Div(id='event-info'),
    dcc.Graph(id='efield-trace'),
    dcc.Graph(id='time-trace'),
    dcc.Graph(id='time-traces'),
    dcc.Graph(id='time-traces2')
])


def get_L1(a):
    ct = np.array(a[1:]) ** 2
    l1 = np.max(ct) / (np.sum(ct) - np.max(ct))
    return l1


@app.callback(
    dash.dependencies.Output('event-info', 'children'),
    [dash.dependencies.Input('event-counter', 'children'),
     dash.dependencies.Input('filename', 'value')],
     [State('user_id', 'children'),
      State('station_id', 'children')])
def update_event_info(evt_counter_json, filename, juser_id, jstation_id):
    print("update event info")
    if filename is None:
        return ""
#     filename = json.loads(jfilename)
    user_id = json.loads(juser_id)
    station_id = json.loads(jstation_id)

    evt_counter = json.loads(evt_counter_json)['evt_counter']
    ariio = provider.get_arianna_io(user_id, filename)
    number_of_events = ariio.get_n_events()
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_stations()[0]
    event_info = ""
    event_info += "{}/{} run {:d} event {:d} {}".format(evt_counter + 1, number_of_events, evt.get_run_number(), evt.get_id(), station.get_station_time())
    if(station.has_parameter(stnp.zenith_cr_templatefit)):
        event_info += "\n cr template fit: {:.1f}, {:.1f}, chi2 = {:.1f}".format(station[stnp.zenith_cr_templatefit] / units.deg,
                                                                                 station[stnp.azimuth_cr_templatefit] / units.deg,
                                                                                 station[stnp.chi2_cr_templatefit])
    if(station.has_parameter(stnp.zenith_nu_templatefit)):
        event_info += "   nu template fit: {:.1f}, {:.1f}, chi2 = {:.1f}".format(station[stnp.zenith_nu_templatefit] / units.deg,
                                                                                 station[stnp.azimuth_nu_templatefit] / units.deg,
                                                                                 station[stnp.chi2_nu_templatefit])
    if(station.has_parameter(stnp.zenith)):
        event_info += "   x-corr fit fit: {:.1f}, {:.1f}".format(station[stnp.zenith] / units.deg,
                                                                 station[stnp.azimuth] / units.deg)
    if(station.has_parameter(stnp.polarization_angle)):
        pol = station[stnp.polarization_angle]
        event_info += "\n pol = {:.1f}".format(pol / units.deg)
        pol = np.abs(pol)
        if(pol > 0.5 * np.pi):
            pol = np.pi - pol
        pol_exp = station[stnp.polarization_angle_expectation]
        event_info += " pol exp = {:.1f}".format(pol_exp / units.deg)
        pol_exp = np.abs(pol_exp)
        if(pol_exp > 0.5 * np.pi):
                pol_exp = np.pi - pol_exp
        event_info += " delta polarization = {:.1f}deg".format(np.abs(pol - pol_exp) / units.deg)
    return event_info


@app.callback(
    dash.dependencies.Output('efield-trace', 'figure'),
    [dash.dependencies.Input('trigger-trace', 'children'),
     dash.dependencies.Input('event-counter', 'children'),
     dash.dependencies.Input('filename', 'value')],
     [State('user_id', 'children'),
      State('station_id', 'children')])
def update_time_efieldtrace(trigger, evt_counter_json, filename, juser_id, jstation_id):
    if filename is None:
        return {}
    print("update efield trace")
#     filename = json.loads(jfilename)
    user_id = json.loads(juser_id)
    station_id = json.loads(jstation_id)

    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    evt_counter = json.loads(evt_counter_json)['evt_counter']
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_stations()[0]
    fig = tools.make_subplots(rows=1, cols=2)
    fig.append_trace(go.Scatter(
            x=station.get_times() / units.ns,
            y=station.get_trace()[0] / units.mV * units.m,
            # text=df_by_continent['country'],
            # mode='markers',
            opacity=0.7,
            marker={
                'color': colors[0],
                'line': {'color': colors[0]}
            },
            name='eR'
        ), 1, 1)
    fig.append_trace(go.Scatter(
            x=station.get_times() / units.ns,
            y=station.get_trace()[1] / units.mV * units.m,
            # text=df_by_continent['country'],
            # mode='markers',
            opacity=0.7,
            marker={
                'color': colors[1],
                'line': {'color': colors[1]}
            },
            name='eTheta'
        ), 1, 1)
    fig.append_trace(go.Scatter(
            x=station.get_times() / units.ns,
            y=station.get_trace()[2] / units.mV * units.m,
            # text=df_by_continent['country'],
            # mode='markers',
            opacity=0.7,
            marker={
                'color': colors[2],
                'line': {'color': colors[2]}
            },
            name='ePhi'
        ), 1, 1)
    fig['layout']['xaxis1'].update(title='time [ns]')
    fig['layout']['yaxis1'].update(title='efield [mV/m]')

    fig.append_trace(go.Scatter(
            x=station.get_frequencies() / units.MHz,
            y=np.abs(station.get_frequency_spectrum()[1]) / units.mV,
            opacity=0.7,
            marker={
                'color': colors[1],
                'line': {'color': colors[1]}
            },
            name='eTheta'
        ), 1, 2)
    fig.append_trace(go.Scatter(
            x=station.get_frequencies() / units.MHz,
            y=np.abs(station.get_frequency_spectrum()[2]) / units.mV,
            opacity=0.7,
            marker={
                'color': colors[2],
                'line': {'color': colors[2]}
            },
            name='ePhi'
        ), 1, 2)
    fig['layout']['xaxis2'].update(title='frequency [MHz]')
    fig['layout']['yaxis2'].update(title='amplitude [mV/m]')
    fig['layout'].showlegend = True
    return fig


@app.callback(
    dash.dependencies.Output('time-trace', 'figure'),
    [dash.dependencies.Input('trigger-trace', 'children'),
     dash.dependencies.Input('event-counter', 'children'),
     dash.dependencies.Input('filename', 'value')],
     [State('user_id', 'children'),
      State('station_id', 'children')])
def update_time_trace(trigger, evt_counter_json, filename, juser_id, jstation_id):
    if filename is None:
        return {}
    print("update time trace")
#     filename = json.loads(jfilename)
    user_id = json.loads(juser_id)
    station_id = json.loads(jstation_id)

    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    evt_counter = json.loads(evt_counter_json)['evt_counter']
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_stations()[0]
    traces = []
    fig = tools.make_subplots(rows=1, cols=2)
    for i, channel in enumerate(station.get_channels()):
        fig.append_trace(go.Scatter(
                x=channel.get_times() / units.ns,
                y=channel.get_trace() / units.mV,
                # text=df_by_continent['country'],
                # mode='markers',
                opacity=0.7,
                marker={
                    'color': colors[i % len(colors)],
                    'line': {'color': colors[i % len(colors)]}
                },
                name=i
            ), 1, 1)
    fig['layout']['xaxis1'].update(title='time [ns]')
    fig['layout']['yaxis1'].update(title='voltage [mV]')
    maxL1 = 0
    maxY = 0
    for i, channel in enumerate(station.get_channels()):
        tt = channel.get_times()
        dt = tt[1] - tt[0]
        trace = channel.get_trace()
        ff = np.fft.rfftfreq(len(tt), dt)
        spec = np.abs(np.fft.rfft(trace, norm='ortho'))
        maxL1 = max(maxL1, get_L1(spec))
        maxY = max(maxY, spec.max())
        fig.append_trace(go.Scatter(
                x=ff / units.MHz,
                y=spec / units.mV,
                opacity=0.7,
                marker={
                    'color': colors[i % len(colors)],
                    'line': {'color': colors[i % len(colors)]}
                },
                name=i
            ), 1, 2)
    fig.append_trace(
           go.Scatter(
                x=[0.9 * ff.max() / units.MHz],
                y=[0.8 * maxY / units.mV],
                mode='text',
                text=['max L1 = {:.2f}'.format(maxL1)],
                textposition='top center'
            ),
        1, 2)
    fig['layout']['xaxis2'].update(title='frequency [MHz]')
    fig['layout']['yaxis2'].update(title='amplitude [mV]')
    fig['layout'].showlegend = False
    return fig


@app.callback(
    dash.dependencies.Output('time-traces', 'figure'),
    [dash.dependencies.Input('event-counter', 'children'),
     dash.dependencies.Input('filename', 'value'),
     dash.dependencies.Input('dropdown-traces', 'value'),
     dash.dependencies.Input('dropdown-trace-info', 'value')],
     [State('user_id', 'children'),
      State('station_id', 'children')])
def update_time_traces(evt_counter_json, filename, dropdown_traces, dropdown_info, juser_id, jstation_id):
#     filename = json.loads(jfilename)
    if filename is None:
        return {}
    user_id = json.loads(juser_id)
    station_id = json.loads(jstation_id)

    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    evt_counter = json.loads(evt_counter_json)['evt_counter']
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_stations()[0]
    traces = []
    fig = tools.make_subplots(rows=len(station.get_channels()), cols=2,
                              shared_xaxes=True, shared_yaxes=False,
                              vertical_spacing=0.01)
    ymax = 0
    if 'trace' in dropdown_traces:
        for i, channel in enumerate(station.get_channels()):
            trace = channel.get_trace() / units.mV
            ymax = max(ymax, np.max(np.abs(trace)))
        for i, channel in enumerate(station.get_channels()):
            tt = channel.get_times() / units.ns
            trace = channel.get_trace() / units.mV
            fig.append_trace(go.Scatter(
                    x=tt,
                    y=trace,
                    # text=df_by_continent['country'],
                    # mode='markers',
                    opacity=0.7,
                    marker={
                        'color': colors[i % len(colors)],
                        'line': {'color': colors[i % len(colors)]}
                    },
                    name=i
                ), i + 1, 1)
            if 'RMS' in dropdown_info:
                fig.append_trace(
                       go.Scatter(
                            x=[0.99 * tt.max()],
                            y=[0.98 * trace.max()],
                            mode='text',
                            text=[r'mu = {:.2f}, STD={:.2f}'.format(np.mean(trace), np.std(trace))],
                            textposition='bottom left'
                        ),
                    i + 1, 1)
    if 'envelope' in dropdown_traces:
        for i, channel in enumerate(station.get_channels()):
            trace = channel.get_trace() / units.mV
            from scipy import signal
            yy = np.abs(signal.hilbert(trace))
            fig.append_trace(go.Scatter(
                    x=channel.get_times() / units.ns,
                    y=yy,
                    # text=df_by_continent['country'],
                    # mode='markers',
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
    if 'crtemplate' in dropdown_traces:
        ref_template = template_provider.get_cr_ref_template(station_id)
        if(station.has_parameter('number_of_templates')):
            ref_templates = template_provider.get_set_of_cr_templates(station_id, n=station['number_of_templates'])
        for i, channel in enumerate(station.get_channels()):
            if(channel.has_parameter('cr_ref_xcorr_template')):
                key = channel['cr_ref_xcorr_template']
                logger.info("using template {}".format(key))
                print(ref_templates.keys())
                ref_template = ref_templates[key][channel.get_id()]
            times = channel.get_times()
            trace = channel.get_trace()
            xcorr = channel['cr_ref_xcorr']
            xcorrpos = channel['cr_ref_xcorr_time']
            dt = times[1] - times[0]
            xcorr_max = xcorr
            if(channel.has_parameter('cr_ref_xcorr_template')):
                xcorr_max = channel['cr_ref_xcorr_max']
            flip = np.sign(xcorr_max)
#             flip = 1
            tttemp = np.arange(0, len(ref_template) * dt, dt)
            yy = flip * np.roll(ref_template * np.abs(trace).max(), int(np.round(xcorrpos / dt)))
            fig.append_trace(go.Scatter(
                    x=tttemp[:len(trace)] / units.ns,
                    y=yy[:len(trace)] / units.mV,
                    # text=df_by_continent['country'],
                    # mode='markers',
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
            fig.append_trace(
               go.Scatter(
                    x=[0.9 * times.max() / units.ns],
                    y=[0.8 * trace.max() / units.mV],
                    mode='text',
                    text=['cr xcorr= {:.2f}, {:.02f}'.format(xcorr, xcorr_max)],
                    textposition='top center'
                ),
            i + 1, 1)
            fig.append_trace(
               go.Scatter(
                    x=[0.9 * times.max() / units.ns],
                    y=[0.5 * trace.max() / units.mV],
                    mode='text',
                    text=['t = {:.0f}ns'.format(xcorrpos)],
                    textposition='top center'
                ),
            i + 1, 1)
    if 'nutemplate' in dropdown_traces:
        ref_template = template_provider.get_nu_ref_template(station_id)
        for i, channel in enumerate(station.get_channels()):
            times = channel.get_times()
            trace = channel.get_trace()
            xcorr = channel['nu_ref_xcorr']
            xcorrpos = channel['nu_ref_xcorr_time']
            dt = times[1] - times[0]
            flip = np.sign(xcorr)
            tttemp = np.arange(0, len(ref_template) * dt, dt)
            yy = flip * np.roll(ref_template * np.abs(trace).max(), int(np.round(xcorrpos / dt)))
            fig.append_trace(go.Scatter(
                    x=tttemp[:len(trace)] / units.ns,
                    y=yy[:len(trace)] / units.mV,
                    # text=df_by_continent['country'],
                    # mode='markers',
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
            fig.append_trace(
               go.Scatter(
                    x=[0.9 * times.max() / units.ns],
                    y=[0.8 * trace.max() / units.mV],
                    mode='text',
                    text=['nu xcorr= {:.2f}'.format(xcorr)],
                    textposition='top center'
                ),
            i + 1, 1)
            fig.append_trace(
               go.Scatter(
                    x=[0.9 * times.max() / units.ns],
                    y=[0.5 * trace.max() / units.mV],
                    mode='text',
                    text=['t = {:.0f}ns'.format(xcorrpos)],
                    textposition='top center'
                ),
            i + 1, 1)
    fig['layout']['xaxis1'].update(title='time [ns]')
    fig['layout']['yaxis1'].update(title='voltage [mV]')
    for i, channel in enumerate(station.get_channels()):
        fig['layout']['yaxis{:d}'.format(i * 2 + 1)].update(range=[-ymax, ymax])

        tt = channel.get_times()
        dt = tt[1] - tt[0]
        trace = channel.get_trace()
        ff = np.fft.rfftfreq(len(tt), dt)
        spec = np.abs(np.fft.rfft(trace, norm='ortho'))
        fig.append_trace(go.Scatter(
                x=ff / units.MHz,
                y=spec / units.mV,
                opacity=0.7,
                marker={
                    'color': colors[i % len(colors)],
                    'line': {'color': colors[i % len(colors)]}
                },
                name=i
            ), i + 1, 2)
        if 'L1' in dropdown_info:
            fig.append_trace(
                   go.Scatter(
                        x=[0.9 * ff.max() / units.MHz],
                        y=[0.8 * spec.max() / units.mV],
                        mode='text',
                        text=['max L1 = {:.2f}'.format(get_L1(spec))],
                        textposition='top center'
                    ),
                i + 1, 2)
    fig['layout'].update(height=1000)
    fig['layout'].showlegend = False
    return fig


@app.callback(
    dash.dependencies.Output('time-traces2', 'figure'),
    [dash.dependencies.Input('event-counter', 'children'),
     dash.dependencies.Input('filename', 'value'),
     dash.dependencies.Input('dropdown-traces', 'value')],
     [State('user_id', 'children'),
      State('station_id', 'children')])
def update_time_traces2(evt_counter_json, filename, dropdown_traces, juser_id, jstation_id):
    if filename is None:
        return {}
#     filename = json.loads(jfilename)
    user_id = json.loads(juser_id)
    station_id = json.loads(jstation_id)

    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    evt_counter = json.loads(evt_counter_json)['evt_counter']
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_stations()[0]
    traces = []
    fig = tools.make_subplots(rows=4, cols=1,
                              shared_xaxes=True, shared_yaxes=True,
                              vertical_spacing=0.05)
    if 'trace' in dropdown_traces:
        for i, iCh in enumerate(range(4, min([8, len(station.get_channels())]))):
            channel = station.get_channels()[iCh]
            fig.append_trace(go.Scatter(
                    x=channel.get_times() / units.ns,
                    y=channel.get_trace() / units.mV,
                    # text=df_by_continent['country'],
                    # mode='markers',
                    opacity=0.7,
                    marker={
                        'color': colors[i % len(colors)],
                        'line': {'color': colors[i % len(colors)]}
                    },
                    name=i
                ), i + 1, 1)
    if 'crtemplate' in dropdown_traces:
        ref_templates = template_provider.get_cr_ref_template(station_id)
        for i, iCh in enumerate(range(4, 8)):
            channel = station.get_channels()[iCh]
            times = channel.get_times()
            trace = channel.get_trace()
            xcorr = channel['cr_ref_xcorr']
            xcorrpos = channel['cr_ref_xcorr_time']
            dt = times[1] - times[0]
            flip = np.sign(xcorr)
            tttemp = np.arange(0, len(ref_templates[channel.get_id()]) * dt, dt)
            yy = flip * np.roll(ref_templates[channel.get_id()] * np.abs(trace).max(), int(np.round(xcorrpos / dt)))
            fig.append_trace(go.Scatter(
                    x=tttemp[:len(trace)] / units.ns,
                    y=yy[:len(trace)] / units.mV,
                    # text=df_by_continent['country'],
                    # mode='markers',
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
    yrange = [-100, 100]
    fig['layout']['xaxis1'].update(title='time [ns]')
    fig['layout']['yaxis1'].update(title='voltage [mV]', range=yrange)
    fig['layout']['yaxis2'].update(title='voltage [mV]', range=yrange)
    fig['layout']['yaxis3'].update(title='voltage [mV]', range=yrange)
    fig['layout']['yaxis4'].update(title='voltage [mV]', range=yrange)
    fig['layout'].update(height=700, width=700)
    fig['layout'].showlegend = False
    return fig
