import dash
from flask import Flask, send_from_directory
import os

server = Flask(__name__, static_folder='static')
app = dash.Dash(server=server)


# @server.route('/favicon.ico')
# def favicon():
#     return send_from_directory(os.path.join(server.root_path, 'static'),
#                                'favicon.ico', mimetype='image/vnd.microsoft.icon')


app.config.suppress_callback_exceptions = True
