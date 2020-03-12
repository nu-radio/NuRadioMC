import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import apps
import apps.add_surface_board
import apps.menu
import apps.add_DRAB
from app import app

app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/add_surface_board':
        return apps.add_surface_board.layout
    if pathname == '/apps/add_DRAB':
        return apps.add_DRAB.layout
    elif pathname == "/apps/menu":
        return apps.menu.layout
    else:
        return "404"


if __name__ == '__main__':
    app.run_server(debug=True)
