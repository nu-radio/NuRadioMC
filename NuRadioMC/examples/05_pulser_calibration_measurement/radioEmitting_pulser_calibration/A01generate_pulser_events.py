import numpy as np
import os
from NuRadioReco.utilities import units
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
    attributes['simulation_mode'] = "emitter"  # must be specified to work for radio emittimg models 
    attributes['n_events'] = n_events  # the number of events contained in this file
    attributes['start_event_id'] = 0  

    ########### FOR Emitter ###############
    data_sets = {}
    data_sets["emitter_antenna_type"] = ["bicone_v8_inf_n1.78"] * n_events
    data_sets["emitter_model"] = ["square"] * n_events
    data_sets["emitter_amplitudes"] = np.ones(n_events) * 1 * units.V

    # we also have choice for the half width and frequency
    data_sets["emitter_half_width"]= 1.0 * np.ones(n_events) *units.ns        # this will be the width of square and tone_burst signal  
    data_sets["emitter_frequency"] = 0.3 * np.ones(n_events)  *units.GHz       # this will be frequency of a signal ( for cw and tone_burst model)
    
    #the position of the emitter
    data_sets["xx"] = np.ones(n_events)* 500 * units.m
    data_sets["yy"] = np.ones(n_events) * 0 * units.m
    data_sets["zz"] = -np.ones(n_events) * 180 * units.m
    
    # the orientation of the emiting antenna, defined via two vectors that are defined with two angles each (see https://nu-radio.github.io/NuRadioReco/pages/detector_database_fields.html)
    # the following definition specifies a traditional “upright” dipole.
    data_sets["emitter_orientation_phi"] = np.ones(n_events) * 0
    data_sets["emitter_orientation_theta"] = np.ones(n_events) * 0
    data_sets["emitter_rotation_phi"] = np.ones(n_events) * 0
    data_sets["emitter_rotation_theta"] = np.ones(n_events) * 90 * units.deg

    # the following informations not particularly useful for radio_emitter models

    data_sets["event_group_ids"] = np.arange(n_events)
    data_sets["shower_ids"] = np.arange(n_events) 
    
    
    ####### For neutrino shower #####
    # the direction of the shower
    # data_sets["azimuths"] = np.zeros(n_events)
    # data_sets["zeniths"] = np.zeros(n_events)

    # everything below this line are required to run NuRadioMC simulation

    # data_sets["shower_type"] = ['had'] * n_events
    # data_sets["shower_energies"] = np.ones(n_events)
    # data_sets["interaction_type"] = np.full(n_events, "nc", dtype='U2')    #for neutrino interactions can be either CC or NC.
    # data_sets["n_interaction"] = np.ones(n_events, dtype=int)
    # data_sets["flavors"] = 12 * np.ones(n_events, dtype=int)    
    # data_sets["energies"] = np.ones(n_events) * 1 * units.EeV
    # data_sets["inelasticity"] = np.ones(n_events)

    
    # write events to file
    write_events_to_hdf5(filename, data_sets, attributes)

if __name__ == "__main__":
    generate_my_events("emitter_event_list.hdf5", 5)

