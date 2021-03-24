Event Display
===========================

We use Dash and plot.ly as a foundation, so you need to
`install it first. <https://dash.plot.ly/installation>`_.

The eventbrowser is a web application, meaning that you can use your webbrowser
to browse through the NuRadioReco events. You start the eventbrowser
(in the subfolder eventbrowser) with
``python index.py /path/to/folder/containing/nur/data/files``.
If all dependencies are installed correctly you should see: ::

    Running on http://127.0.0.1:8080/ (Press CTRL+C to quit)
    Restarting with inotify reloader
    Debugger is active!
    Debugger PIN: 224-474-503

Then just open http://127.0.0.1:8080/ in your favorite web browser and select the
file you want to see.

Alternatively, you can use the file *NuRadioViewer* in the *eventbrowser* directory.
Add the directory to you system *$PATH* and you can open the eventbrowser by
typing ``NuRadioViewer filename.nur``.
