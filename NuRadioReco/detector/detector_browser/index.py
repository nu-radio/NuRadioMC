
import dash_html_components as html
import dash
from NuRadioReco.detector.detector_browser import open_file
from NuRadioReco.detector.detector_browser import detector_map
from NuRadioReco.detector.detector_browser import station_info
from NuRadioReco.detector.detector_browser import channel_info
from NuRadioReco.detector.detector_browser import hardware_response
import argparse
import os
from NuRadioReco.detector.detector_browser.app import app

argparser = argparse.ArgumentParser(description="Visualization for the detector")
argparser.add_argument('file_location', type=str, help="Path of folder or filename.")
parsed_args = argparser.parse_args()
data_folder = os.path.dirname(parsed_args.file_location)

app.title = 'Detector Browser'

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div(id='folder-dummy', style={'display': 'none'}, children=data_folder),
            open_file.layout,
            station_info.layout,
            channel_info.layout
        ], style={'flex': '1'}),
        html.Div([
            detector_map.layout,
            hardware_response.layout
        ], style={'flex': '2'})
    ], style={'display': 'flex'})
])

if __name__ == '__main__':
    if int(dash.__version__.split('.')[0]) <= 1:
        if int(dash.__version__.split('.')[1]) < 0:
            print('WARNING: Dash version 0.39.0 or newer is required, you are running version {}. Please update.'.format(dash.__version__))
    app.run_server(debug=False, port=8080)
