import dash
import json
import plotly
import numpy as np
from app import app
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.eventbrowser.apps.common
import dataprovider
provider = dataprovider.DataProvider()

layout = [
    dcc.RadioItems(
        id='station-overview-rec-sim',
        options=[
            {'label': 'Reconstruction', 'value': 'rec'},
            {'label': 'Simulation', 'value': 'sim'}
        ],
        value='rec'
    ),
    html.Div(id='station-overview-properties')
]

station_properties_for_overview = [
    {
        'label': 'Zenith [deg]',
        'param': stnp.zenith,
        'unit': units.deg
    },{
        'label': 'Azimuth [deg]',
        'param': stnp.azimuth,
        'unit': units.deg
    },{
        'label': 'Neutrino Energy [eV]',
        'param': stnp.nu_energy,
        'unit': units.eV
    },{
        'label': 'Cosmic Ray Energy [eV]',
        'param': stnp.cr_energy,
        'unit': units.eV
    }
]
@app.callback(Output('station-overview-properties', 'children'),
                [Input('filename', 'value'),
                Input('event-counter-slider', 'value'),
                Input('station-id-dropdown', 'value'),
                Input('station-overview-rec-sim', 'value')],
                [State('user_id', 'children')])
def station_overview_properties(filename, evt_counter, station_id, rec_or_sim, juser_id):
    if filename is None or station_id is None:
        return ''
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    if station is None:
        return []
    if rec_or_sim == 'rec':
        prop_station = station
    else:
        if station.get_sim_station() is None:
            return []
        else:
            prop_station = station.get_sim_station()
    if prop_station.is_neutrino():
        event_type = 'Neutrino'
    elif prop_station.is_cosmic_ray():
        event_type = 'Cosmic Ray'
    else:
        event_type = 'Unknown'
    reply = []
    reply.append(
        html.Div([
            html.Div('Event Type:', className='custom-table-td'),
            html.Div(str(event_type), className='custom-table-td custom-table-td-last')
        ], className='custom-table-row')
    )
    props = NuRadioReco.eventbrowser.apps.common.get_properties_divs(prop_station, station_properties_for_overview)
    for prop in props:
        reply.append(prop)
    return reply
