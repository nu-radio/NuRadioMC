import numpy as np
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
    attributes['n_events'] = n_events  # the number of events contained in this file
    attributes['start_event_id'] = 0
    # define the fiducial simulation volume. Instead of specifying fiducial_rmin and fiducial_rmin one can also specify
    # fiducial_xmin, fiducial_xmax, fiducial_ymin and fiducial_ymax
    # the concept of the diducial volume is described in the NuRadioMC paper. In short: only interactions in this smaller
    # fiducial volume are saved. This is useful for the simulation of secondary interactions. For this dummy example
    # the fiduial volume is the same as the full volume.
    attributes['fiducial_rmin'] = 0
    attributes['fiducial_rmax'] = 1 * units.km
    attributes['fiducial_zmin'] = 0 * units.m
    attributes['fiducial_zmax'] = -2 * units.km
    # define the full simulation volume. Instead of specifying rmin and rmin one can also specify
    # xmin, xmax, ymin and ymax
    attributes['rmin'] = 0
    attributes['rmax'] = 1 * units.km
    attributes['zmin'] = 0 * units.m
    attributes['zmax'] = -2 * units.km

    attributes['volume'] = attributes['rmax'] ** 2 * np.pi * np.abs(attributes['zmax'])

    # if only interactions on a surface (e.g. for muons from air showers) are generated, the surface area needs to be
    # specified attributes['area']

    # define the minumum and maximum energy
    attributes['Emin'] = 1 * units.EeV
    attributes['Emax'] = 1 * units.EeV

    # the interval of zenith directions
    attributes['thetamin'] = 0
    attributes['thetamax'] = np.pi
    # the interval of azimuths directions
    attributes['phimin'] = 0
    attributes['phimax'] = 2 * np.pi

    # now generate the events and fill all required data sets
    # here we fill all data sets with dummy values

    # each line/entry specified a particle shower of certain energy.
    # In principle only the shower direction (zeniths, azimuths fields), the shower position (xx, yy, zz fields)
    # the shower energy, the shower type and the event group id are required. We set them first
    data_sets = {}
    # the direction of the shower
    data_sets["azimuths"] = np.ones(n_events)
    data_sets["zeniths"] = np.ones(n_events)
    # the position of the shower
    data_sets["xx"] = np.ones(n_events)
    data_sets["yy"] = np.ones(n_events)
    data_sets["zz"] = np.ones(n_events)
    # the shower energy
    data_sets["shower_energies"] = np.ones(n_events) * 1 * units.EeV
    # the shower type (here we only generate hadronic showers). This infomration is needed for the Askaryan emission model
    data_sets["shower_type"] = ['had'] * n_events
    # give each shower a unique id (we can also have multiple showers for a single event by just giving several showers
    # the same event_group_id)
    data_sets["event_group_ids"] = np.arange(n_events)
    data_sets["shower_ids"] = np.arange(n_events)

    # there are a couple of additional parameters that are required to run a NuRadioMC simulations. These parameters
    # don't influence the simulated radio signals but are required for other post analysis tasks. If these parameters
    # are not relevant for the type of data you're generating, just set them to any value.

    # specify which interaction it is (only relevant if multiple showers from the same initial neutrino are simulated)
    # here it is just 1 for all events.
    data_sets["n_interaction"] = np.ones(n_events, dtype=int)

    # the neutrino flavor. Here we only generate electron neutinos which have the integer code 12.
    # the neutrino flavor is only used in the calculation of the "weight", i.e. the probability of the neutrino reaching
    # the detector. If other particles than a neutrino are simulated, just set the flavor to the corresponding particle code
    # following https://pdg.lbl.gov/2019/reviews/rpp2019-rev-monte-carlo-numbering.pdf or just set it to zero.
    data_sets["flavors"] = 12 * np.ones(n_events, dtype=int)
    # the neutrino energy. This field is also only used for the weight calculation.
    data_sets["energies"] = np.ones(n_events) * 1 * units.EeV

    # optionally one can also directly set the event weight here (useful if particles other than neutrinos, or calibration
    # setups are simulated
    # data_sets["weights"] = np.ones(n_events)

    # the interaction type. For neutrino interactions is can be either CC or NC. This parameter is not used but passed
    # to the output file for information purposes.
    data_sets["interaction_type"] = np.full(n_events, "nc", dtype='U2')
    # The inelasiticiy, i.e. the fraction of the neutrino energy that is transferred into the hadronic shower.
    # This parameter is not used but saved into the output file for information purposes.
    data_sets["inelasticity"] = np.ones(n_events)

    # write events to file
    write_events_to_hdf5(filename, data_sets, attributes)


# add some test code
if __name__ == "__main__":
    generate_my_events("testfile.hdf5", 20)

