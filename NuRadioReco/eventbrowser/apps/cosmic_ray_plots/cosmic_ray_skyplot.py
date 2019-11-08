import dash
import json
import plotly
from NuRadioReco.utilities import units
from NuRadioReco.eventbrowser.default_layout import default_layout
import numpy as np
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app
import dataprovider
provider = dataprovider.DataProvider()

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
    ariio = provider.get_arianna_io(user_id, filename)
    traces = []
    keys = ariio.get_header()[station_id].keys()
    if stnp.cr_zenith in keys and stnp.cr_azimuth in keys:
        traces.append(plotly.graph_objs.Scatterpolar(
            r=np.rad2deg(ariio.get_header()[station_id][stnp.cr_zenith]),
            theta=np.rad2deg(ariio.get_header()[station_id][stnp.cr_azimuth]),
            text=[str(x) for x in ariio.get_event_ids()],
            mode='markers',
            name='cosmic ray events',
            opacity=1,
            customdata=[x for x in range(ariio.get_n_events())],
            marker=dict(
                color='blue'
            )
        ))
    if current_selection != []:
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
