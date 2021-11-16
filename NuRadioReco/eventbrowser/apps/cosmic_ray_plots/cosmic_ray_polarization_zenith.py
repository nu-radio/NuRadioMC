import json
import plotly
from NuRadioReco.utilities import units
import numpy as np
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from NuRadioReco.eventbrowser.app import app
import NuRadioReco.eventbrowser.dataprovider
provider = NuRadioReco.eventbrowser.dataprovider.DataProvider()

layout = [
    html.Div([
        html.Div([
            dcc.Graph(id='cr-polarization-zenith')
        ], style={'flex': '1'}),
    ], style={'display': 'flex'})
]


@app.callback(Output('cr-polarization-zenith', 'figure'),
              [Input('filename', 'value'),
               Input('btn-open-file', 'value'),
               Input('event-ids', 'children'),
               Input('station-id-dropdown', 'value')],
              [State('user_id', 'children')])
def plot_cr_polarization_zenith(filename, btn, jcurrent_selection, station_id, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    nurio = provider.get_file_handler(user_id, filename)
    traces = []
    pol = []
    pol_exp = []
    zeniths = []
    for i_event in range(nurio.get_n_events()):
        event = nurio.get_event_i(i_event)
        for station in event.get_stations():
            for electric_field in station.get_electric_fields():
                if electric_field.has_parameter(efp.polarization_angle) and electric_field.has_parameter(efp.polarization_angle_expectation) and electric_field.has_parameter(efp.zenith):
                    pol.append(electric_field.get_parameter(efp.polarization_angle))
                    pol_exp.append(electric_field.get_parameter(efp.polarization_angle_expectation))
                    zeniths.append(electric_field.get_parameter(efp.zenith))
    pol = np.array(pol)
    pol = np.abs(pol)
    pol[pol > 0.5 * np.pi] = np.pi - pol[pol > 0.5 * np.pi]
    pol_exp = np.array(pol_exp)
    pol_exp = np.abs(pol_exp)
    pol_exp[pol_exp > 0.5 * np.pi] = np.pi - pol_exp[pol_exp > 0.5 * np.pi]
    zeniths = np.array(zeniths)
    traces.append(plotly.graph_objs.Scatter(
        x=zeniths / units.deg,
        y=np.abs(pol - pol_exp) / units.deg,
        text=[str(x) for x in nurio.get_event_ids()],
        mode='markers',
        customdata=[x for x in range(nurio.get_n_events())],
        opacity=1
    ))

    current_selection = json.loads(jcurrent_selection)
    if current_selection:
        for trace in traces:
            trace['selectedpoints'] = current_selection

    return {
        'data': traces,
        'layout': plotly.graph_objs.Layout(
            xaxis={'type': 'linear', 'title': 'zenith angle [deg]'},
            yaxis={'title': 'polarization angle error [deg]', 'range': [0, 90]},
            hovermode='closest'
        )
    }
