import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from NuRadioReco.detector.webinterface.apps import add_surface_board
from NuRadioReco.detector.webinterface.apps import menu
from NuRadioReco.detector.webinterface.apps import add_DRAB
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
    if pathname == '/apps/add_DRAB':
        return add_DRAB.layout
    elif pathname == "/apps/menu":
        return menu.layout
    else:
        return "404"


if __name__ == '__main__':
    app.run_server(debug=True)
