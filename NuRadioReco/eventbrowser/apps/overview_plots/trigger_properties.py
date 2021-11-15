import json
from NuRadioReco.eventbrowser.app import app
from dash import html
import numpy as np
from dash.dependencies import Input, Output, State
import NuRadioReco.eventbrowser.dataprovider

provider = NuRadioReco.eventbrowser.dataprovider.DataProvider()

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
    nurio = provider.get_file_handler(user_id, filename)
    evt = nurio.get_event_i(evt_counter)
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
            display_value = '{}'
            setting_value = trigger.get_trigger_settings()[setting_name]
            if type(setting_value) in [float, np.float32, np.float64, np.float128]:
                display_value = '{:.5g}'
            props.append(
                html.Div([
                    html.Div('{}'.format(setting_name), className='custom-table-td'),
                    html.Div(display_value.format(setting_value),
                             className='custom-table-td custom-table-td-last')
                ], className='custom-table-row')
            )
        reply.append(html.Div(props))
    return reply
