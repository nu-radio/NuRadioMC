import dash
from flask import Flask
import os

# if environment variable is set, use this instead of __name__ for the app location,
# this helps combining the eventbrowser app with RNOGDataViewer
server = Flask(os.getenv("FLASK_APP_DIR") or __name__, static_folder='static')
app = dash.Dash(server=server)


# @server.route('/favicon.ico')
# def favicon():
#     return send_from_directory(os.path.join(server.root_path, 'static'),
#                                'favicon.ico', mimetype='image/vnd.microsoft.icon')


app.config.suppress_callback_exceptions = True
