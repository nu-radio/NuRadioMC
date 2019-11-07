import dash
import json
import plotly
from NuRadioReco.utilities import units
from NuRadioReco.eventbrowser.default_layout import default_layout
import numpy as np
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp

def plot_cr_polarization_zenith(filename, btn, jcurrent_selection, station_id, juser_id, provider):
    if filename is None or station_id is None:
        return {}
    user_id = json.loads(juser_id)
    ariio = provider.get_arianna_io(user_id, filename)
    traces = []
    keys = ariio.get_header()[station_id].keys()
    pol = []
    pol_exp = []
    zeniths = []
    for i_event in range(ariio.get_n_events()):
        event = ariio.get_event_i(i_event)
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
        text=[str(x) for x in ariio.get_event_ids()],
        mode='markers',
        customdata=[x for x in range(ariio.get_n_events())],
        opacity=1
    ))

    current_selection = json.loads(jcurrent_selection)
    if current_selection != []:
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
