import json
import numpy as np
from NuRadioReco.utilities import units

"""
This file generates the positions for the RNO array as drawn in the upcoming
white paper. If the distance between stations wants to be changed, the user
can change the default side argument in the RNO_array function.

To run it, just type:
python generate_RNO.py

A json file called station_rest.json with the station coordinates will be created.
This can be combined with a channels description to form a detector file.
"""

def RNO_array(side=1.*units.km, N_x=6, N_y=7):

    pos_x = np.linspace(-float(N_x)/2*side, float(N_x)/2*side, N_x)
    pos_y = np.linspace(-float(N_y)/2*side, float(N_y)/2*side, N_y)

    forbidden = [(0,0), (3,0), (4,0), (4,1), (5,0), (5,1), (5,6)]

    station = {}
    iS = 1
    station_id = 101
    for i_x, x in enumerate(pos_x):
        for i_y, y in enumerate(pos_y):

            if (i_x, i_y) in forbidden:
                continue

            key = '{:02d}'.format(iS)
            station[key] = {}
            station[key]["pos_northing"] = x
            station[key]["pos_easting"] = y
            station[key]["station_id"] = station_id
            station_id += 1
            iS += 1

    return station

station = RNO_array()
with open('station_rest.json', 'w') as fout:
    json.dump(station, fout, sort_keys=True, indent=4)
