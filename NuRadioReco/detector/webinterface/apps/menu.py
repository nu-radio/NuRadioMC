import dash_core_components as dcc
import dash_html_components as html

layout = html.Div([
    html.H3('Welcome to the RNO-G Hardware Databse Uploader'),
    dcc.Link('Add S parameter measurement of Surface Board', href='/apps/add_surface_board'),
    html.Br(),
    dcc.Link('Add DRAB unit measurement', href='/apps/add_DRAB'),
    html.Br(),
    dcc.Link('Add S21 Cable measurments', href='/apps/add_CABLE'),
    html.Br(),
    dcc.Link('Add S21 SURFACE Cable measurments', href='/apps/add_surf_CABLE'),
    html.Br(),
    dcc.Link('Add S11 VPol measurment', href='/apps/add_VPol'),
    html.Br(),
    dcc.Link('Add S parameter measurement of IGLU board', href='/apps/add_IGLU'),
    html.Br(),
    dcc.Link('Add Pulser measurement (placeholder)', href='/apps/add_Pulser'),
    html.Br(),
    dcc.Link('Add HPol measurement (placeholder)', href='/apps/add_HPol'),
    html.Br(),
    dcc.Link('Add DAQ measurement (placeholder)', href='/apps/add_DAQ'),
    html.Br(),
    dcc.Link('Read and download calibration values', href='/apps/reader')
])
