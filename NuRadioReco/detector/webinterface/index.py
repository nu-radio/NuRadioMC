import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from NuRadioReco.detector.webinterface.apps import add_surface_board
from NuRadioReco.detector.webinterface.apps import menu
from NuRadioReco.detector.webinterface.apps import add_DRAB
from NuRadioReco.detector.webinterface.apps import add_IGLU
from NuRadioReco.detector.webinterface.apps import add_VPol
from NuRadioReco.detector.webinterface.apps import add_CABLE
from NuRadioReco.detector.webinterface.apps import add_surf_CABLE
from NuRadioReco.detector.webinterface.apps import add_Pulser
from NuRadioReco.detector.webinterface.apps import add_HPol
from NuRadioReco.detector.webinterface.apps import add_DAQ
from NuRadioReco.detector.webinterface.apps import downhole_chain
from NuRadioReco.detector.webinterface.apps import reader
from NuRadioReco.detector.webinterface.apps import channel
from NuRadioReco.detector.webinterface.apps import station
from NuRadioReco.detector.webinterface.app import app

# app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/add_surface_board':
        return add_surface_board.layout
    elif pathname == '/apps/downhole_chain':
        return downhole_chain.layout
    elif pathname == '/apps/add_DRAB':
        return add_DRAB.layout
    elif pathname == '/apps/add_IGLU':
        return add_IGLU.layout
    elif pathname == '/apps/add_CABLE':
        return add_CABLE.layout
    elif pathname == '/apps/add_surf_CABLE':
        return add_surf_CABLE.layout
    elif pathname == '/apps/add_VPol':
        return add_VPol.layout
    elif pathname == '/apps/add_Pulser':
        return add_Pulser.layout
    elif pathname == '/apps/add_HPol':
        return add_HPol.layout
    elif pathname == '/apps/add_DAQ':
        return add_DAQ.layout
    elif pathname == '/apps/reader':
        return reader.layout
    elif pathname == '/apps/channel':
        return channel.layout
    elif pathname == '/apps/station':
        return station.layout
    else:
        return menu.layout


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)
    #for running locally switch with comment below
    #app.run_server(debug=True)
