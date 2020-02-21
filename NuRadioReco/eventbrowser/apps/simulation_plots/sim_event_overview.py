import dash
import json
import plotly
from NuRadioReco.utilities import units
from NuRadioReco.eventbrowser.default_layout import default_layout
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp
import radiotools.helper as hp
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app
import dataprovider
provider = dataprovider.DataProvider()

layout = [
    dcc.Graph(id='sim-event-3d', style={'flex': '1'}),
    html.Div([
        dcc.Dropdown(id='sim-station-properties-dropdown', options=[], multi=True, value=[]),
        html.Div(id='sim-station-properties-table', className='table table-striped')
    ], style={'flex': '1', 'min-height': '500px'})
]

@app.callback(
    Output('sim-event-3d', 'figure'),
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value'),
    Input('station-id-dropdown', 'value')],
    [State('user_id', 'children')]
     )
def update_sim_event_3d(i_event, filename, station_id, juser_id):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_station(station_id)
    sim_station = station.get_sim_station()
    if sim_station is None:
        return {}
    data = [plotly.graph_objs.Scatter3d(
        x = [0],
        y = [0],
        z = [0],
        mode = 'markers',
        name = 'Station'
        )]
    if sim_station.has_parameter(stnp.nu_vertex):
        vertex = sim_station.get_parameter(stnp.nu_vertex)
        data.append(plotly.graph_objs.Scatter3d(
            x = [vertex[0]],
            y = [vertex[1]],
            z = [vertex[2]],
            mode = 'markers',
            name = 'Interaction Vertex'
            ))
        plot_range = 1.5*np.max(np.abs(vertex))
        if sim_station.has_parameter(stnp.nu_zenith) and sim_station.has_parameter(stnp.nu_azimuth):
            neutrino_path = hp.spherical_to_cartesian(sim_station.get_parameter(stnp.nu_zenith), sim_station.get_parameter(stnp.nu_azimuth))
            data.append(plotly.graph_objs.Scatter3d(
                x = [vertex[0], vertex[0] + .25*plot_range*neutrino_path[0]],
                y = [vertex[1], vertex[1] + .25*plot_range*neutrino_path[1]],
                z = [vertex[2], vertex[2] + .25*plot_range*neutrino_path[2]],
                name = 'Neutrino Direction',
                mode = 'lines'
                ))
    else:
        plot_range = 1*units.km
    fig = plotly.graph_objs.Figure(
        data = data,
            layout = plotly.graph_objs.Layout(
                width = 1000,
                height = 1000,
                legend = {
                    'orientation': 'h',
                    'y': 1.1
                },
                scene = {
                'aspectmode': 'manual',
                'aspectratio': {
                    'x': 2,
                    'y': 2,
                    'z': 1
                },
                'xaxis' : {
                    'range': [-plot_range,plot_range]
                },
                'yaxis' : {
                    'range': [-plot_range, plot_range]
                },
                'zaxis' : {
                    'range': [-plot_range,0]
                }
                }
            )
        )
    return fig

@app.callback(Output('sim-station-properties-dropdown', 'options'),
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value'),
    Input('station-id-dropdown', 'value')],
    [State('user_id', 'children')])
def get_sim_station_property_options(i_event, filename, station_id, juser_id):
    if filename is None or station_id is None:
        return []
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_station(station_id).get_sim_station()
    if station is None:
        return []
    options = []
    for parameter in stnp:
        if station.has_parameter(parameter):
            options.append({'label': parameter.name, 'value': parameter.value})
    return options
@app.callback(Output('sim-station-properties-table', 'children'),
    [Input('event-counter-slider', 'value'),
    Input('filename', 'value'),
    Input('sim-station-properties-dropdown', 'value'),
    Input('station-id-dropdown', 'value')],
    [State('user_id', 'children')])
def get_sim_station_property_table(i_event, filename, properties, station_id, juser_id):
    if filename is None or station_id is None:
         return []
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(i_event)
    station = evt.get_station(station_id).get_sim_station()
    reply = []
    for property in properties:
        reply.append(
            html.Div([
                html.Div(str(stnp(property).name), className='custom-table-td'),
                html.Div(str(station.get_parameter(stnp(property))), className='custom-table-td custom-table-td-last')
            ],className='custom-table-row')
        )
    return reply
