from NuRadioReco.modules.base.module import register_run
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import hashlib

from NuRadioReco.utilities import units
from NuRadioReco.framework.hybrid_shower import HybridShower
from NuRadioReco.framework.parameters import stationParameters as stp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.utilities.LOFAR import LORA_CORE_PRECISION, LORA_ANGLE_PRECISION

logger = logging.getLogger("NuRadioReco.LOFAR.LORASimulator")


class LORASimulator:
    """
    Simulates the LORA particle detectors to get a rough estimate of the shower core position and energy. This is a simplified version which does not include all the complexities of the actual LORA detectors. But in the future, it should be included.

    author: Keito Watanabe

    """

    def __init__(self, log_level = logging.INFO):
        self.__debug = False
        self.__random_seed = None
        logger.setLevel(log_level)

    def begin(self, debug=False, random_seed=None):
        """
        Initialize the LORA particle detector simulator.

        Parameters:
        -----------
        debug : bool, default=False
            debug flag
        random_seed : int, default=None
            the random seed to set the core uncertainty.

            The default is None, which will use the numpy default value. 
        """
        self.__debug = debug
        self.__random_seed = random_seed

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
        true_primary_energy = coreas_shower.get_parameter(shp.energy)
        mag_field = coreas_shower.get_parameter(shp.magnetic_field_vector)

        max_retries = 2
        retries = 0
        triggered = False

        attempted_cores_list = []

        while not triggered and retries < max_retries:
            retries += 1

            rand_x, rand_y = None, None

            if rand_x is None or rand_y is None:
                rng = np.random.default_rng(seed=self.__random_seed)
                rand_x = true_core[0] + rng.normal(0, LORA_CORE_PRECISION)
                rand_y = true_core[1] + rng.normal(0, LORA_CORE_PRECISION)
                logger.info(
                    f"Generated reproducible core guess (seed={self.__random_seed}): x={rand_x:.2f}, y={rand_y:.2f}"
                )

            attempted_cores_list.append((rand_x, rand_y))

        core_guess = np.array([rand_x * units.m, rand_y * units.m, 7.6 * units.m])  # z is fixed to 7.6 m, which is the average height of the LORA detectors
        # add warning if any of the x or y core positions are larger than 100 m, since the core reconstruction uncertainty diminishes beyond this
        # TODO: in future, characterise this behaviour and inject this into this module instead
        if np.logical_or(np.abs(rand_x - true_core[0]) > 100 * units.m, np.abs(rand_y - true_core[1]) > 100 * units.m):
            logger.warning(f"Core ({rand_x / units.m:.1f}, {rand_y / units.m:.1f}) m is greater than 100m. Data events with this core will not perform well. Proceed with caution.")
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

        # generate a hybrid shower containing this information. This will be the particle shower
        lora_shower = HybridShower("LORA")
        lora_shower.set_parameter(shp.zenith, zenith_guess)
        lora_shower.set_parameter(shp.azimuth, azimuth_guess)
        lora_shower.set_parameter(shp.core, core_guess)
        lora_shower.set_parameter(shp.energy, true_primary_energy)
        lora_shower.set_parameter(shp.magnetic_field_vector, mag_field)

        # add the lora shower to the event
        event.get_hybrid_information().add_hybrid_shower(lora_shower)

        return lora_shower
    
    def end(self):
        """
        End the LORA simulation module. This function can be used to perform any cleanup or finalization tasks after the simulation is complete.
        """
        logger.info("LORA simulation module has completed.")

        