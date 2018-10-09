import h5py
import matplotlib.pyplot as plt
import argparse
import numpy as np
from utilities import units

parser = argparse.ArgumentParser(description='plot electric field traces at detector stations')
parser.add_argument('inputfilename', type=str,
                    help='path to (hdf5) input file')

args = parser.parse_args()

fin = h5py.File(args.inputfilename, 'r')
for event_name, event in fin.iteritems():
    event_id = event.attrs['event_id']  # event id is stored as an attribute of the event data set

    for station_name, station in event.iteritems():
        position = station.attrs['position']
        t, Ex, Ey, Ez = np.array(station).T

        fig, ax = plt.subplots(1, 1)
        ax.plot(t / units.ns, Ex / units.V * units.m, label='Ex')
        ax.plot(t / units.ns, Ey / units.V * units.m, label='Ey')
        ax.plot(t / units.ns, Ez / units.V * units.m, label='Ez')
        ax.set_title('{} at position {}'.format(station_name, position))
        ax.set_xlabel('time [ns]')
        ax.set_ylabel('amplitude [V/m]')
        ax.legend()
        plt.show()

