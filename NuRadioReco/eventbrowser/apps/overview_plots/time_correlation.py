import dash
import json
import plotly
from NuRadioReco.utilities import units
from NuRadioReco.eventbrowser.default_layout import default_layout
import numpy as np
from NuRadioReco.framework.parameters import stationParameters as stnp

def plot_corr(xcorr_type, filename, jcurrent_selection, station_id, event_type, juser_id, provider):
    if filename is None or station_id is None or xcorr_type is None:
        return {}
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    fig = plotly.subplots.make_subplots(rows=1, cols=1)
    keys = ariio.get_header()[station_id].keys()
    if event_type == 'nu':
        if not stnp.nu_xcorrelations in keys:
            return {}
        xcorrs = ariio.get_header()[station_id][stnp.nu_xcorrelations]
    else:
        if not stnp.cr_xcorrelations in keys:
            return {}
        xcorrs = ariio.get_header()[station_id][stnp.cr_xcorrelations]
    if stnp.station_time in keys:
        times = []
        for time in ariio.get_header()[station_id][stnp.station_time]:
            times.append(time.value)
        fig.append_trace(plotly.graph_objs.Scatter(
            x=times,
            y=[xcorrs[i][xcorr_type] for i in range(len(xcorrs))],
            text=[str(x) for x in ariio.get_event_ids()],
            customdata=[x for x in range(ariio.get_n_events())],
            mode='markers',
            opacity=1
        ),1,1)
    else:
        return {}
    current_selection = json.loads(jcurrent_selection)
    if current_selection != []:
        for trace in traces:
            trace['selectedpoints'] = current_selection
    fig['layout'].update(default_layout)
    fig['layout']['yaxis'].update({'title': xcorr_type, 'range': [0, 1]})
    fig['layout']['hovermode'] = 'closest'
    return fig
