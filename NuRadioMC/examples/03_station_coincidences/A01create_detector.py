import copy
import json
from matplotlib import pyplot as plt
import numpy as np
from NuRadioReco.utilities import units
from radiotools import plthelpers as php

with open("single_position.json") as fin:
    detector_single = json.load(fin)

    detector_full ={}
    detector_full['stations'] = detector_single['stations']
    detector_full['channels'] = {}
    # insert station at center
    i = -1
    for channel in detector_single['channels'].values():
        i += 1
        channel = copy.copy(channel)
        channel['channel_id'] = i
        detector_full['channels'][str(i+1)] = channel

    distances = [100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000]
    xx = [0]
    yy = [0]
    for d in distances:
        for x in [-d, 0, d]:
            for y in [-d, 0, d]:
                if(x == 0 and y == 0):
                    continue
                for channel in detector_single['channels'].values():
                    i += 1
                    channel = copy.copy(channel)
                    channel['ant_position_x'] += (x)
                    channel['ant_position_y'] += (y)
                    channel['channel_id'] = i
                    detector_full['channels'][str(i+1)] = channel
                    xx.append(x)
                    yy.append(y)

    with open('horizontal_spacing_detector.json', 'w') as fout:
        json.dump(detector_full, fout, indent=4, separators=(',', ': '))

    xx = np.array(xx)
    yy = np.array(yy)
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    for i, x in enumerate(np.unique(np.abs(xx))):
        # select all stations corresponding to this distance
        mask = ((np.abs(xx) == x) & (np.abs(yy) == x))  \
            | ((np.abs(xx) == x) & (yy == 0)) \
            | ((np.abs(yy) == x) & (xx == 0))
        ax.plot(xx[mask], yy[mask], php.get_marker2(i))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.savefig("layout.pdf", bbox='tight')
    plt.show()
