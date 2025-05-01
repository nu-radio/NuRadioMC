from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioReco.utilities import units
from six import iterkeys, iteritems
from scipy import constants
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import h5py
from NuRadioMC.EvtGen.generator import write_events_to_hdf5
import logging
logger = logging.getLogger("NuRadioMC.EventGen")

VERSION_MAJOR = 1
VERSION_MINOR = 1


def generate_my_events(filename, n_events):
    """
    Event generator skeleton

    Parameters
    ----------
    filename: string
        the output filename of the hdf5 file
    n_events: int
        number of events to generate
    """

    # first set the meta attributes
    attributes = {}
    n_events = int(n_events)
    attributes['n_events'] = n_events
    attributes['start_event_id'] = 0
    attributes['simulation_mode'] = 'emitter'

    # now generate the events and fill all required data sets
    # here we fill all data sets with dummy values
    data_sets = {}

    # define the emitter positions. X/Y are the easting/northing coordinates of the SPICE core (https://journals.aps.org/prd/pdf/10.1103/PhysRevD.105.123012)
    data_sets["xx"] = np.ones(n_events) * 12911 * units.m
    data_sets["yy"] = np.ones(n_events) * 14927.3 * units.m
    data_sets["zz"] = -np.linspace(0, 2000, n_events) * units.m

    data_sets["event_group_ids"] = np.arange(n_events)
    data_sets["shower_ids"] = np.arange(n_events)

    data_sets['emitter_model'] = np.array(['efield_idl1_spice'] * n_events)
    data_sets['emitter_amplitudes'] = np.ones(n_events) # for the efield_idl1_spice model, this quantity is the relative rescaling of the measured amplitudes

    write_events_to_hdf5(filename, data_sets, attributes)

if __name__ == "__main__":
    generate_my_events("test_input.hdf5", 10)
