import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from NuRadioReco.detector.webinterface.apps import add_surface_board
from NuRadioReco.detector.webinterface.apps import menu
from NuRadioReco.detector.webinterface.apps import add_DRAB
from NuRadioReco.detector.webinterface.apps import add_IGLO
from NuRadioReco.detector.webinterface.apps import add_VPol
from NuRadioReco.detector.webinterface.apps import add_CABLE
from NuRadioReco.detector.webinterface.apps import add_surf_CABLE
from NuRadioReco.detector.webinterface.apps import add_Pulser
from NuRadioReco.detector.webinterface.apps import add_HPol
from NuRadioReco.detector.webinterface.apps import add_DAQ
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
    elif pathname == '/apps/add_DRAB':
        return add_DRAB.layout
    elif pathname == '/apps/add_IGLO':
        return add_IGLO.layout
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
    else:
        return menu.layout


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)
    #for running locally switch with comment below
    #app.run_server(debug=True)
