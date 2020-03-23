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
logger = logging.getLogger("EventGen")
logging.basicConfig()

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
#     attributes['fiducial_rmin'] = 0
#     attributes['fiducial_rmax'] = 1 * units.km
#     attributes['fiducial_zmin'] = 0 * units.m
#     attributes['fiducial_zmax'] = -2 * units.km
#     attributes['rmin'] = 0
#     attributes['rmax'] = 1 * units.km
#     attributes['zmin'] = 0 * units.m
#     attributes['zmax'] = -2 * units.km
#     attributes['Emin'] = 1 * units.EeV
#     attributes['Emax'] = 1 * units.EeV

    # now generate the events and fill all required data sets
    # here we fill all data sets with dummy values
    data_sets = {}
    # the 'neutrino direction' needs to be set but are irrelevant for the simulation, because we simulate a
    # uniform emitter
    data_sets["azimuths"] = np.ones(n_events)
    data_sets["zeniths"] = np.ones(n_events)

    # define the emitter positions. X/Y are the easting/northing coordinates of the SPICE core
    data_sets["xx"] = np.ones(n_events) * 42600 * units.feet
    data_sets["yy"] = np.ones(n_events) * 48800 * units.feet
    # simualte different depth
    data_sets["zz"] = -np.linspace(0, 1800, n_events) * units.m
    data_sets["event_ids"] = np.arange(n_events)
    data_sets["n_interaction"] = np.ones(n_events, dtype=np.int)
    data_sets["interaction_type"] = np.array(['had'] * n_events)

    # again these parameters are irrelevant for our simulation but still need to be set
    data_sets["flavors"] = np.array([12 for i in range(n_events)])
    data_sets["energies"] = np.ones(n_events) * 1 * units.eV
    data_sets["inelasticity"] = np.ones(n_events) * 0.5

    # write events to file
    write_events_to_hdf5(filename, data_sets, attributes)


if __name__ == "__main__":
    generate_my_events("input_spice.hdf5", 1000)
