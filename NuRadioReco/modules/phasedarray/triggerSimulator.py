from NuRadioReco.utilities import units
from NuRadioReco.framework.trigger import SimplePhasedTrigger
import numpy as np
import time
import logging
logger = logging.getLogger('phasedTriggerSimulator')

cspeed = constants.c * units.m / units.s

default_angles = np.linspace(-45,47,14) * units.deg


class triggerSimulator:
    """
    Calculates the trigger for a phased array

    explain the module in more detail here
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

    def get_beam_rolls(self, station, phasing_angles=default_angles,
                       sampling_rate=1., ref_index=1.78):

        ant_z = self.get_vertical_positions(station, det, triggered_channels)
        self.check_vertical_string(ant_z)
        beam_rolls = []
        ref_z = np.max(ant_z)

        for angle in phasing_angles:
            subbeam_rolls = {}
            for z, channel_id in zip(ant_z, station.iter_channels):
                delay = -(ant_z-ref_z)/cspeed * ref_index * np.sin(angle)
                roll = int(delay/sampling_rate)
                subbeam_rolls[channel_id] = roll
            beam_rolls.append(subbeam_rolls)

        return beam_rolls

    def check_vertical_string(self, ant_z):

        diff_z = np.array(ant_z) - ant_z[0]
        if ( sum(diff_z) > 1.e-3*units.m ):
            raise NotImplementedError('The phased triggering array should lie on a vertical line')

    def run(self, evt, station, det,
            threshold=60 * units.mV,
            triggered_channels=None,
            trigger_name='simple_phased_threshold',
            phasing_angles=default_angles):
        """
        simulates phased array trigger for each event

        describe the run method in more detail here (if neccessary).
        I left triggered_channels in as an optional parameter, don't know if you
        need it.

        Parameters
        ----------
        triggered_channels: array of ints
            channels ids that are triggered on
        """

        sampling_rate = station.get_channel(0).get_sampling_rate()
        time_step = 1./sampling_rate

        phased_trace = None

    	if (triggered_channels == None):

    		triggered_channels = [channel._id for channel in station.iter_channels()]

        beam_rolls = self.get_beam_rolls(station, ant_z, phasing_angles, sampling_rate, ref_z)

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

    	is_triggered = False
        if(np.max(np.abs(phased_trace)) > threshold):  # define a simple threshold trigger
            is_triggered = True
    	    logger.debug("Station has triggered")

    	trigger = SimplePhasedTrigger(trigger_name, threshold, triggered_channels)
     	trigger.set_triggered(is_triggered)
    	station.set_trigger(trigger)

    def end(self):
        pass
