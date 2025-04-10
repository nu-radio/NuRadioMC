import numpy as np
import os
from NuRadioReco.utilities import units
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
    attributes['simulation_mode'] = "emitter"  # must be specified to work for radio emittimg models 
    attributes['n_events'] = n_events  # the number of events contained in this file
    attributes['start_event_id'] = 0  

    ########### FOR Emitter ###############
    data_sets = {}

    # we allow for the simulation of multiple emitters that emit at the same time. We use the event_group_ids to group them together
    # and the `shower_ids` to distinguish between different emitters within the same event_group, i.e., the same procedure as for
    # particle simulations with several showers per event group.
    # in this example, we only have one emitter per event group.
    data_sets["event_group_ids"] = np.arange(n_events, dtype=int)
    data_sets["shower_ids"] = np.arange(n_events, dtype=int)

    # data_sets["emitter_antenna_type"] = ["bicone_v8_inf_n1.78"] * n_events
    data_sets["emitter_model"] = ["efield_delta_pulse"] * n_events
    data_sets["emitter_amplitudes"] = np.ones(n_events) * 10 * units.V

    # we also have choice for the half width and frequency
    data_sets["emitter_polarization"]= 0.0 * np.ones(n_events) # this will be the polarization of the signal (for delta_pulse model, 0=eTheta)
    data_sets["emitter_half_width"]= 1.0 * np.ones(n_events) *units.ns        # this will be the width of square and tone_burst signal  
    data_sets["emitter_frequency"] = 0.3 * np.ones(n_events)  *units.GHz       # this will be frequency of a signal ( for cw and tone_burst model)
    data_sets["emitter_polarization"] = 0.1 * np.ones(n_events)  # this will be the polarization of the signal (for delta_pulse model, 0=eTheta)


    #the position of the emitter
    data_sets["xx"] = np.ones(n_events) * 42600 * units.feet
    data_sets["yy"] = np.ones(n_events) * 48800 * units.feet
    data_sets["zz"] = np.linspace(-400, -1600, n_events)

    # the orientation of the emiting antenna, defined via two vectors that are defined with two angles each (see https://nu-radio.github.io/NuRadioReco/pages/detector_database_fields.html)
    # the following definition specifies a traditional “upright” dipole.
    data_sets["emitter_orientation_phi"] = np.ones(n_events) * 0
    data_sets["emitter_orientation_theta"] = np.ones(n_events) * 0
    data_sets["emitter_rotation_phi"] = np.ones(n_events) * 0
    data_sets["emitter_rotation_theta"] = np.ones(n_events) * 90 * units.deg



    # write events to file
    write_events_to_hdf5(filename, data_sets, attributes)

if __name__ == "__main__":
    generate_my_events("SPICE_drop_event_list.hdf5", 10)

