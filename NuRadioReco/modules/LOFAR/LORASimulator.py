from NuRadioReco.modules.base.module import register_run
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import hashlib

from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.utilities.LOFAR import LORA_CORE_PRECISION, LORA_ANGLE_PRECISION

logger = logging.getLogger("NuRadioReco.LOFAR.LORASimulator")


class LORASimulator:
    """
    Simulates the LORA particle detectors to get a rough estimate of the shower core position and energy. This is a simplified version which does not include all the complexities of the actual LORA detectors. But in the future, it should be included.

    author: Keito Watanabe

    """

    def __init__(self):
        self.__debug = False

    def begin(self, debug=False):
        self.__debug = debug

    @register_run()
    def run(self, event, det):
        """
        
        Incorporates the LORA particle detectors to get a rough estimate of the shower core position and energy.

        Currently, we only take a normal distribution around the average LORA core position, energy, and arrival direction uncertainty. But in principle this should depend on e.g. the detector position relative to the antennas, and also would be per event (possibly). 

        Parameters
        ----------
        event: Event object
            The event to which the LORA simulation should be applied.
        station: Station object
            The station whose channels noise shall be added to
        det: Detector object
            The detector description

        """
        # extract the true shower parameters from the event
        coreas_shower = event.get_first_sim_shower()
        true_zenith = coreas_shower.get_parameter(shp.zenith)
        true_azimuth = coreas_shower.get_parameter(shp.azimuth)
        true_core = coreas_shower.get_parameter(shp.core)

        max_retries = 2
        retries = 0
        triggered = False

        attempted_cores_list = []

        while not triggered and retries < max_retries:
            retries += 1

            rand_x, rand_y = None, None

            if rand_x is None or rand_y is None:
                stem_hash = int(hashlib.md5(event.get_id().encode()).hexdigest(), 16) % (
                    2**32
                )
                rng = np.random.default_rng(seed=stem_hash)
                rand_x = true_core[0] + rng.normal(0, LORA_CORE_PRECISION)
                rand_y = true_core[1] + rng.normal(0, LORA_CORE_PRECISION)
                logger.info(
                    f"Generated reproducible core guess (seed={stem_hash}): x={rand_x:.2f}, y={rand_y:.2f}"
                )

            attempted_cores_list.append((rand_x, rand_y))

        core_guess = np.array([rand_x * units.m, rand_y * units.m, 0.0])
        logger.info(
            f"Generated core guess: x={core_guess[0]:.2f}, y={core_guess[1]:.2f}"
        )

        # Draw a single LORA-like direction estimate (truth + LORA angular error).
        # This guessed direction -- NOT the truth -- is what the voltage->E-field
        # unfolding below uses, exactly as in real data, so the recovered fluences
        # carry the same direction-induced error the data pipeline has.
        azimuth_guess = true_azimuth + rng.normal(0.0, LORA_ANGLE_PRECISION)
        zenith_guess = true_zenith + rng.normal(0.0, LORA_ANGLE_PRECISION)
        logger.info(
            f"Generated direction guess: zenith={np.degrees(zenith_guess):.2f} deg, azimuth={np.degrees(azimuth_guess):.2f} deg"
        )

        # now add the guessed core and direction to each STATION (not sim_station, because that is the truth) in the event, so that the unfolding uses this guessed direction and core.
        for station in event.get_stations():
            station.set_parameter(shp.zenith, zenith_guess)
            station.set_parameter(shp.azimuth, azimuth_guess)
            station.set_parameter(shp.core, core_guess)
        logger.info(
            f"Added guessed core and direction to {len(event.get_stations())} stations in the event."
        )

        return core_guess
    
    def end(self):
        """
        End the LORA simulation module. This function can be used to perform any cleanup or finalization tasks after the simulation is complete.
        """
        logger.info("LORA simulation module has completed.")

        