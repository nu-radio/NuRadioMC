import dash
import json
import plotly
from NuRadioReco.utilities import units
from NuRadioReco.eventbrowser.default_layout import default_layout
import numpy as np
from NuRadioReco.framework.parameters import stationParameters as stnp
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app
import dataprovider
provider = dataprovider.DataProvider()

layout = [
    html.Div(id='trigger', style={'display': 'none'},
             children=json.dumps(None)),
    html.Div([
        dcc.Graph(id='skyplot-xcorr')
    ], className='row'),
    html.Div(id='output')
]

@app.callback(Output('skyplot-xcorr', 'figure'),
              [Input('filename', 'value'),
               Input('trigger', 'children'),
               Input('event-ids', 'children'),
               Input('station-id-dropdown', 'value')],
              [State('user_id', 'children')])
def plot_rec_directions(filename, trigger, jcurrent_selection, station_id, juser_id):
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
