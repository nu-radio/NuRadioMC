import dash
import json
import plotly
import numpy as np
from app import app
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from NuRadioReco.utilities import units
import NuRadioReco.eventbrowser.apps.common
import dataprovider
provider = dataprovider.DataProvider()

layout = [html.Div(id='trigger-overview-properties')]

@app.callback(Output('trigger-overview-properties', 'children'),
                [Input('filename', 'value'),
                Input('event-counter-slider', 'value'),
                Input('station-id-dropdown', 'value')],
                [State('user_id', 'children')])
def trigger_overview_properties(filename, evt_counter, station_id, juser_id):
    if filename is None or station_id is None:
        return ''
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    evt = ariio.get_event_i(evt_counter)
    station = evt.get_station(station_id)
    if station is None:
        return []
    reply = []
    for trigger_name in station.get_triggers():
        props = [
        html.Div([
            html.Div('{}'.format(trigger_name), className='custom-table-th')
        ], className='custom-table-row')
        ]
        trigger = station.get_trigger(trigger_name)
        for setting_name in trigger.get_trigger_settings():
            props.append(
                html.Div([
                    html.Div('{}'.format(setting_name), className='custom-table-td'),
                    html.Div('{}'.format(trigger.get_trigger_settings()[setting_name]), className='custom-table-td custom-table-td-last')
                ], className='custom-table-row')
            )
        reply.append(html.Div(props))
    return reply
