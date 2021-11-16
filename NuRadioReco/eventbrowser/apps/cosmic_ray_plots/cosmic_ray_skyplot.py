import json
import plotly
import numpy as np
from NuRadioReco.framework.parameters import stationParameters as stnp
from dash import dcc
from dash.dependencies import Input, Output, State
from NuRadioReco.eventbrowser.app import app
import NuRadioReco.eventbrowser.dataprovider
provider = NuRadioReco.eventbrowser.dataprovider.DataProvider()

layout = [
    dcc.Graph(id='cr-skyplot'),
]


@app.callback(Output('cr-skyplot', 'figure'),
              [Input('filename', 'value'),
               Input('trigger', 'children'),
               Input('event-ids', 'children'),
               Input('btn-open-file', 'value'),
               Input('station-id-dropdown', 'value')],
              [State('user_id', 'children')])
def cosmic_ray_skyplot(filename, trigger, jcurrent_selection, btn, station_id, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    current_selection = json.loads(jcurrent_selection)
    nurio = provider.get_file_handler(user_id, filename)
    traces = []
    keys = nurio.get_header()[station_id].keys()
    if stnp.cr_zenith in keys and stnp.cr_azimuth in keys:
        traces.append(plotly.graph_objs.Scatterpolar(
            r=np.rad2deg(nurio.get_header()[station_id][stnp.cr_zenith]),
            theta=np.rad2deg(nurio.get_header()[station_id][stnp.cr_azimuth]),
            text=[str(x) for x in nurio.get_event_ids()],
            mode='markers',
            name='cosmic ray events',
            opacity=1,
            customdata=[x for x in range(nurio.get_n_events())],
            marker=dict(
                color='blue'
            )
        ))
    if current_selection:
        for trace in traces:
            trace['selectedpoints'] = current_selection
    return {
        'data': traces,
        'layout': plotly.graph_objs.Layout(
            showlegend=True,
            hovermode='closest',
            height=500
        )
    }
