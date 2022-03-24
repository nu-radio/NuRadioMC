from dash import dcc
from dash import html
import base64

test_png = 'rno_logo.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')


layout = html.Div([
    html.H3('Welcome to the RNO-G Hardware Databse Uploader'),
    html.Img(src='data:image/png;base64,{}'.format(test_base64), style={'float':'right', 'width':'50%'}),
    dcc.Link('Add FULL DOWNHOLE CHAIN measurement', href='/apps/downhole_chain'),
    html.Br(),
    html.Br(),
    dcc.Link('Add SURFACE Board S Parameters', href='/apps/add_surface_board'),
    html.Br(),
    html.Br(),
    dcc.Link('Add DRAB S Parameters', href='/apps/add_DRAB'),
    html.Br(),
    html.Br(),
    dcc.Link('Add IGLU Board S Parameters', href='/apps/add_IGLU'),
    html.Br(),
    html.Br(),
    dcc.Link('Add Downhole Fiber Cable measurments (S21)', href='/apps/add_CABLE'),
    html.Br(),
    html.Br(),
    dcc.Link('Add Surface Channel Cable measurments (S21)', href='/apps/add_surf_CABLE'),
    html.Br(),
    html.Br(),
    dcc.Link('Add VPol measurment (S11)', href='/apps/add_VPol'),
    html.Br(),
    html.Br(),
    dcc.Link('Add Pulser measurement (placeholder)', href='/apps/add_Pulser'),
    html.Br(),
    html.Br(),
    dcc.Link('Add HPol measurement (placeholder)', href='/apps/add_HPol'),
    html.Br(),
    html.Br(),
    dcc.Link('Add DAQ measurement (placeholder)', href='/apps/add_DAQ'),
    html.Br(),
    html.Br(),
    dcc.Link('Read and download calibration values', href='/apps/reader'),
    html.Br(),
    html.Br(),
    dcc.Link('Insert Station or Channel details', href='/apps/channel'),
    html.Br(),
    html.Br(),
    dcc.Link('Build a Station', href='/apps/station')])
