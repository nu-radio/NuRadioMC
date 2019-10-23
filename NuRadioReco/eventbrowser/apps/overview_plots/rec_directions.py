import dash
import json
import plotly
from NuRadioReco.utilities import units
from NuRadioReco.eventbrowser.default_layout import default_layout
import numpy as np
from NuRadioReco.framework.parameters import stationParameters as stnp

def plot_rec_directions(filename, trigger, jcurrent_selection, station_id, juser_id, provider):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    current_selection = json.loads(jcurrent_selection)
    ariio = provider.get_arianna_io(user_id, filename)
    traces = []
    keys = ariio.get_header()[station_id].keys()
    if stnp.zenith in keys and stnp.azimuth in keys:
        traces.append(plotly.graph_objs.Scatterpolar(
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
        'layout': plotly.graph_objs.Layout(
            showlegend= True,
            hovermode='closest',
            height=500
        )
    }
