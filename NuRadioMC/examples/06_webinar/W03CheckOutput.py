from NuRadioMC.utilities.plotting import plot_vertex_distribution
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
import os

parser = argparse.ArgumentParser(description='Check NuRadioMC output')
parser.add_argument('--filename', type=str, default='results/out.hdf5',
                    help='path to NuRadioMC simulation output')
args = parser.parse_args()

filename = args.filename

fin = h5py.File(filename, 'r')

print(fin.keys())
print(fin.attrs.keys())
print(fin['station_101'].keys())
#print(list(fin['station_101']['multiple_triggers']))

print(fin.attrs['trigger_names'])
trigger_names = np.array(fin.attrs['trigger_names'])
trigger_index = np.squeeze(np.argwhere(trigger_names == 'hilo_2of4_5_sigma'))
mask_simple_trigger = np.array(fin['station_101']['multiple_triggers'])[:,1]
mask_coinc_trigger = np.array(fin['station_101']['multiple_triggers'])[:,trigger_index]
xx = np.array(fin['xx'])[mask_coinc_trigger]
yy = np.array(fin['yy'])[mask_coinc_trigger]
zz = np.array(fin['zz'])[mask_coinc_trigger]
weights = np.array(fin['weights'])[mask_coinc_trigger]
fig, ax = plot_vertex_distribution(xx, yy, zz,
                                   weights=np.array(weights),
                                   rmax=fin.attrs['rmax'],
                                   zmin=fin.attrs['zmin'])
plt.show()
