from NuRadioReco.utilities import units
from NuRadioReco.framework.trigger import SimplePhasedTrigger
import numpy as np
from scipy import constants
import time
import logging
logger = logging.getLogger('phasedTriggerSimulator')

cspeed = constants.c * units.m / units.s

main_low_angle = -53. * units.deg
main_high_angle = 47 * units.deg
default_angles = np.arcsin( np.linspace( np.sin(main_low_angle), np.sin(main_high_angle), 15) )

sec_low_angle = main_low_angle/2
sec_high_angle = main_high_angle/2
default_sec_angles = np.arcsin(np.linspace( np.sin(sec_low_angle), np.sin(sec_high_angle), 15) )


class triggerSimulator:
    """
    Calculates the trigger for a phased array with a primary and a secondary beam.
    The channels that participate in both beams and the pointing angle for each
    subbeam can be specified.

    See https://arxiv.org/pdf/1809.04573.pdf
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self, debug=False, pre_trigger_time=100 * units.ns):
        self.__pre_trigger_time = pre_trigger_time
        self.__debug = debug

    def get_vertical_positions(self, station, det, triggered_channels=None):

        ant_z = [ det.get_relative_position(station.get_id(), channel.get_id())[2] \
                    for channel in station.iter_channels() \
                    if channel.get_id() in triggered_channels ]

        return np.array(ant_z)

    def get_beam_rolls(self, station, det, triggered_channels,
                       phasing_angles=default_angles, ref_index=1.78):
        """
        Calculates the delays needed for phasing the array.
        """
        sampling_rate = station.get_channel(0).get_sampling_rate()
        time_step = 1./sampling_rate

        ant_z = self.get_vertical_positions(station, det, triggered_channels)
        self.check_vertical_string(ant_z)
        beam_rolls = []
        ref_z = (np.max(ant_z)+np.min(ant_z))/2

        for angle in phasing_angles:
            subbeam_rolls = {}
            for z, channel_id in zip(ant_z, triggered_channels):
                delay = (z-ref_z)/cspeed * ref_index * np.sin(angle)
                roll = int(delay/time_step)
                subbeam_rolls[channel_id] = roll
            logger.debug("angle:", angle/units.deg)
            logger.debug(subbeam_rolls)
            beam_rolls.append(subbeam_rolls)

        return beam_rolls

    def check_vertical_string(self, ant_z):

        diff_z = np.array(ant_z) - ant_z[0]
        diff_z = np.abs(diff_z)
        if ( sum(diff_z) > 1.e-3*units.m ):
            raise NotImplementedError('The phased triggering array should lie on a vertical line')

    def run(self, evt, station, det,
            threshold=60 * units.mV,
            triggered_channels=None,
            secondary_channels=None,
            trigger_name='simple_phased_threshold',
            phasing_angles=default_angles,
            secondary_phasing_angles=default_sec_angles):
        """
        simulates phased array trigger for each event

        Several channels are phased by delaying their signals by an amount given
        by a pointing angle. Several pointing angles are possible in order to cover
        the sky. The array triggered_channels controls the channels that are phased,
        according to the angles phasing_angles. A secondary phasing that is added
        to this primary phasing is possible, and it is controlled by the parametres
        secondary_channels and secondary_phasing_angles.

        Parameters
        ----------
        threshold: float
            threshold above (or below) a trigger is issued, absolute amplitude
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        secondary_channels: array of int
            channel ids of the channels that form the secondary phasing array
            if None, only the channels in even indexes are taken
        trigger_name: string
            name for the trigger
        phasing_angles: array of float
            pointing angles for the primary beam
        secondary_phasing_angles: array of float
            pointing angles for the secondary beam
        """

        phased_trace = None

    	if (triggered_channels == None):
    		triggered_channels = [channel._id for channel in station.iter_channels()]

        if (secondary_channels == None):
            secondary_channels = triggered_channels[::2]

        logger.debug("primary channels:", triggered_channels)
        beam_rolls = self.get_beam_rolls(station, det, triggered_channels, phasing_angles)
        logger.debug("secondary_channels:", secondary_channels)
        secondary_beam_rolls = self.get_beam_rolls(station, det, secondary_channels, secondary_phasing_angles)

        for channel in station.iter_channels():  # loop over all channels (i.e. antennas) of the station
            channel_id = channel.get_id()
            if channel_id not in triggered_channels:  # skip all channels that do not participate in the trigger decision
                logger.debug("skipping channel{}".format(channel_id))
                continue

            trace = channel.get_trace()  # get the time trace (i.e. an array of amplitudes)
            #times = channel.get_times()  # get the corresponding time bins

            for subbeam_rolls in beam_rolls:

                if(phased_trace is None):
                    phased_trace = np.roll(trace, subbeam_rolls[channel_id])
                else:
                    phased_trace += np.roll(trace, subbeam_rolls[channel_id])

            if channel_id in secondary_channels:
                for secbeam_rolls in secondary_beam_rolls:
                    phased_trace += np.roll(trace, secbeam_rolls[channel_id])

    	is_triggered = False
        if(np.max(np.abs(phased_trace)) > threshold):  # define a simple threshold trigger
            is_triggered = True
    	    logger.debug("Station has triggered")

    	trigger = SimplePhasedTrigger(trigger_name, threshold, triggered_channels, secondary_channels, phasing_angles, secondary_phasing_angles)
     	trigger.set_triggered(is_triggered)
    	station.set_trigger(trigger)

    def end(self):
        pass
