import dash
from flask import Flask

server = Flask(__name__, static_folder='static')

app = dash.Dash(server=server)
app.config.suppress_callback_exceptions = True
