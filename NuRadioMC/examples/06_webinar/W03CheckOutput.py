from NuRadioMC.utilities.plotting import plot_vertex_distribution
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
import time
import os

"""
This file teaches how to read the output from NuRadioMC simulations. To run it
with the default results/NuMC_output.hdf5 file, just run:

python W03CheckOutput.py

Else, use:

python W03CheckOutput.py --filename path/to/output_file.hdf5
"""

parser = argparse.ArgumentParser(description='Check NuRadioMC output')
parser.add_argument('--filename', type=str, default='results/NuMC_output.hdf5',
                    help='path to NuRadioMC simulation output')
args = parser.parse_args()

if __name__ == "__main__":
    filename = args.filename

    """
    First we open the HDF5 file using the h5py module. When the File function is
    called, it returns a dictionary having HDF5 data sets and groups inside. The way
    to convert them to numpy arrays or lists is just taking these data sets and
    passing them to the numpy.array() or python-native list() functions.
    """
    fin = h5py.File(filename, 'r')

    """
    Let us print all the keys in the fin dictionary. All of these datasets are
    available to be used.
    """
    print("There are the following keys in the file:")
    print("-----------------------------------------")
    for key in fin.keys():
        print(key)
    print("-----------------------------------------")
    time.sleep(2)

    """
    We can also print the attributes that give general information about the simulation,
    not referred to any event in particular.
    """
    print("There are the following keys in the file attributes:")
    print("----------------------------------------------------")
    for key in fin.attrs.keys():
        print(key)
    print("----------------------------------------------------")
    time.sleep(2)

    """
    Some of the keys begin with 'station_'. This means that these keys link to
    groups that contain more data sets inside them. For instance, the key 'station_101'
    contains all the data sets related to the station 101.
    """
    print("There are the following keys in the station_101 group:")
    print("----------------------------------------------------")
    for key in fin['station_101'].keys():
        print(key)
    print("----------------------------------------------------")
    time.sleep(2)

    """
    And the station groups also contain an attribute with the antenna positions
    """
    print("There are the following keys in the station_101 group attributes:")
    print("-----------------------------------------------------------------")
    for key in fin['station_101'].attrs.keys():
        print(key)
    print("-----------------------------------------------------------------")
    time.sleep(2)

    """
    Let us now extract some information from the output file. Let's say we are
    interested in knowing the vertex positions of the events that have triggered
    station 101. First, we should know what triggers this detector has.
    """
    print('This is the list of used triggers for this detector')
    trigger_names = np.array(fin.attrs['trigger_names'])
    print(trigger_names)

    """
    Now that we know the triggers, we can either use the n-th trigger and then retrieve
    their names using the trigger_names list, or we can search what index corresponds
    to the trigger we want. Let's say we want to know the results for the trigger
    'hilo_2of4_5_sigma'. We can find its index in the trigger_names list.
    """
    chosen_trigger = 'hilo_2of4_5_sigma'
    trigger_index = np.squeeze(np.argwhere(trigger_names == chosen_trigger))
    print("The trigger {} corresponds to the index {}".format(chosen_trigger, trigger_index))
    time.sleep(2)

    """
    It should be 0 if we have followed all the steps of the example correctly. Now,
    we would like to create a mask to select only the events that have been triggered
    using the high-low 2 out of 4 trigger. We can access the 'multiple_triggers'
    data set in the 'station_101' group. This data set is a 2-D array, with the first
    dimension refers to the event and the second controls the type of trigger. We can
    create a 1-D mask (an array with bools) containing the trigger information:
    """
    mask_coinc_trigger = np.array(fin['station_101']['multiple_triggers'])[:, trigger_index]

    """
    If instead of station 101 we were interested in the global trigger for the detector,
    containing all of the stations, we could drop the 'station_101' key and just use
    the global fin['multiple_triggers'][:, trigger_index]. In this example, there is
    only one station so the result should be the same.

    Let us now take the vertex positions and use the mask to eliminate from the array
    all the vertices corresponding to non-triggering events. 'xx', 'yy', and 'zz'
    are the data sets containing the vertex positions.
    """
    xx = np.array(fin['xx'])[mask_coinc_trigger]
    yy = np.array(fin['yy'])[mask_coinc_trigger]
    zz = np.array(fin['zz'])[mask_coinc_trigger]

    """
    We also need the statistical weights for each triggering neutrino. Remember that
    each of these weights represents the probability for the neutrino to reach our
    fiducial volume.
    """
    weights = np.array(fin['weights'])[mask_coinc_trigger]

    """
    All right! We have all the information we need. We can plot it with our favourite
    2D histogram tool. We are going to use plot_vertex_distribution, a function that
    can be found in the utilities.plotting module. The results are going to be a bit
    underwhelming because our example contains a handful of events only, but results
    are spectacular for millions of events. And even with a handful of events we are
    able to see the triangle where the events trigger.
    """
    fig, ax = plot_vertex_distribution(xx, yy, zz,
                                    weights=np.array(weights),
                                    rmax=fin.attrs['rmax'],
                                    zmin=fin.attrs['zmin'])
    plt.show()
