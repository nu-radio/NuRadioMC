from __future__ import absolute_import, division, print_function  # , unicode_literals
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import dash
from flask import Flask
from NuRadioReco.detector import detector
import logging
from NuRadioReco.modules.base import module

logger = module.setup_logger(level=logging.INFO)

det = detector.Detector(source='sql')

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Loading screen CSS
# app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
# app.css.append_css({"external_url": "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"})


app.title = 'ARIANNA detector database'

app.layout = html.Div([
    # represents the URL bar, doesn't render anything
    dcc.Location(id='url', refresh=False),

    dcc.Dropdown(id='station_id',
                 options=[{'label': ll, 'value': ll} for ll in [14, 15, 17, 18, 19, 30, 32, 50, 51, 52, 61]]),

    html.Div(id='main')
])

keys = ['channel_id', 'commission_time', 'decommission_time',
        'ant_position_x', 'ant_position_y', 'ant_position_z',
        'ant_type',
        'ant_comment',
        'ant_orientation_phi', 'ant_orientation_theta',
        'ant_rotation_phi', 'ant_rotation_theta',
        'ant_deployment_time',
        'amp_reference_measurement', 'amp_type',
        'cab_id', 'cab_length', 'cab_reference_measurement', 'cab_time_delay', 'cab_type',
        'adc_id', 'adc_n_samples', 'adc_nbits', 'adc_sampling_frequency', 'adc_time_delay',
        ]


# next/previous buttons


@app.callback(
    Output('main', 'children'),
    [Input('station_id', 'value')])
def show_station(station_id):
    if not det.has_station(station_id):
        return html.H2(children="station {} not present in data base".format(station_id))

    tts = det.get_unique_time_periods(station_id)
    d = [html.H1(children="station {}".format(station_id))]
    for iT, t0 in enumerate(tts[:-1]):
        t1 = tts[iT + 1]
        d.append(html.H4(children="station configuration from {} - {}".format(t0, t1)))
        det.update(t0)
        header = [html.Tr([html.Th(col) for col in keys])]

        for iC in det.get_channel_ids(station_id):
            c = det.get_channel(station_id, iC)
            header.append(html.Tr([html.Td(c[key]) for key in keys]))
        d.append(html.Table(header))

    res = html.Div(d)
    return res


if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
