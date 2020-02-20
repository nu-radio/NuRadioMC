import dash
import json
import plotly
import numpy as np
from app import app
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import NuRadioReco.eventbrowser.apps.common
import dataprovider
provider = dataprovider.DataProvider()

layout = [
    dcc.RadioItems(
        id='efield-overview-rec-sim',
        options=[
            {'label': 'Reconstruction', 'value': 'rec'},
            {'label': 'Simulation', 'value': 'sim'}
        ],
        value='rec'
    ),
    html.Div(id='efield-overview-properties')
]

efield_properties_for_overview = [
    {
        'label': 'Ray Path Type',
        'param': efp.ray_path_type,
        'unit': None
    },{
        'label': 'Zenith [deg]',
        'param': efp.zenith,
        'unit': units.deg
    },{
        'label': 'Azimuth [deg]',
        'param': efp.azimuth,
        'unit': units.deg
    },{
        'label': 'spectrum Slope',
        'param': efp.cr_spectrum_slope,
        'unit': None
    },{
        'label': 'Energy Fluence [eV]',
        'param': efp.signal_energy_fluence,
        'unit': units.eV
    },{
        'label': 'Polarization Angle [deg]',
        'param': efp.polarization_angle,
        'unit': units.deg
    },{
        'label': 'Expected Polarization Angle [deg]',
        'param': efp.polarization_angle_expectation,
        'unit': units.deg
    }
]

@app.callback(Output('efield-overview-properties', 'children'),
                [Input('filename', 'value'),
                Input('event-counter-slider', 'value'),
                Input('station-id-dropdown', 'value'),
                Input('efield-overview-rec-sim', 'value')],
                [State('user_id', 'children')])
def efield_overview_properties(filename, evt_counter, station_id, rec_sim, juser_id):
    if filename is None or station_id is None:
        return ''
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    if station is None:
        return []
    reply = []
    if rec_sim == 'rec':
        chosen_station = station
    else:
        if station.get_sim_station() is None:
            return {}
        chosen_station = station.get_sim_station()
    for electric_field in chosen_station.get_electric_fields():
        props = NuRadioReco.eventbrowser.apps.common.get_properties_divs(electric_field, efield_properties_for_overview)
        reply.append(html.Div([
            html.Div('Channels', className='custom-table-td'),
            html.Div('{}'.format(electric_field.get_channel_ids()), className='custom-table-td custom-table-td-last')
        ], className='custom-table-row'))
        reply.append(html.Div(props, style={'margin': '0 0 30px'}))
    return reply
