from NuRadioReco.utilities import units
from NuRadioReco.framework.trigger import SimplePhasedTrigger
import numpy as np
from scipy import constants
import time
import logging
logger = logging.getLogger('phasedTriggerSimulator')

cspeed = constants.c * units.m / units.s

main_low_angle = -53. * units.deg
main_high_angle = 47. * units.deg
default_angles = np.arcsin( np.linspace( np.sin(main_low_angle), np.sin(main_high_angle), 15) )

default_sec_angles = 0.5*(default_angles[:-1]+default_angles[1:])
default_sec_angles = np.insert(default_sec_angles, len(default_sec_angles), default_angles[-1] + 3.5*units.deg)

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

    def get_antenna_positions(self, station, det, triggered_channels=None, component=2):
        """
        Calculates the vertical coordinates of the antennas of the detector
        """

        ant_pos = [ det.get_relative_position(station.get_id(), channel.get_id())[component] \
                    for channel in station.iter_channels() \
                    if channel.get_id() in triggered_channels ]

        return np.array(ant_pos)

    def get_beam_rolls(self, station, det, triggered_channels,
                       phasing_angles=default_angles, ref_index=1.55):
        """
        Calculates the delays needed for phasing the array.
        """
        sampling_rate = station.get_channel(0).get_sampling_rate()
        time_step = 1./sampling_rate

        ant_z = self.get_antenna_positions(station, det, triggered_channels, 2)
        self.check_vertical_string(station, det, triggered_channels)
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

    def check_vertical_string(self, station, det, triggered_channels):
        """
        Checks if the triggering antennas lie in a straight vertical line
        """

        cut = 1.e-3*units.m
        ant_x = self.get_antenna_positions(station, det, triggered_channels, 0)
        diff_x = np.abs(ant_x - ant_x[0])
        ant_y = self.get_antenna_positions(station, det, triggered_channels, 1)
        diff_y = np.abs(ant_y - ant_y[0])
        if ( sum(diff_x) > cut or sum(diff_y) > cut ):
            raise NotImplementedError('The phased triggering array should lie on a vertical line')

    def phased_trigger(self, station, beam_rolls, sec_beam_rolls, triggered_channels, threshold, window_time=10.67*units.ns):
        """
        Calculates the trigger for a certain phasing configuration.
        Beams are formed. A set of overlapping time windows is created and
        the square of the voltage is averaged within each window.
        If the average surpasses the given threshold squared times the
        number of antennas in any of these windows, a trigger is created.
        This is an ARA-like power trigger.

        Parameters
        ----------
        station: Station object
            Description of the current station
        beam_rolls: array of ints
            Contains the integers for rolling the voltage traces (delays)
        sec_beam_rolls: array of ints
            Contains the secondary beam integers for rolling the voltage traces (delays)
        triggered_channels: array of ints
            Ids of the triggering channels
        threshold: float
            Voltage threshold for a SINGLE antenna. It will be rescaled with the
            square root of the number of antennas.
        window_time: float
            Width of the time window used in the power integration

        Returns
        -------
        is_triggered: bool
            True if the triggering condition is met
        """
        sampling_rate = station.get_channel(0).get_sampling_rate()
        time_step = 1./sampling_rate

        for subbeam_rolls, sec_subbeam_rolls in zip(beam_rolls, sec_beam_rolls):

            phased_trace = None
            # Number of antennas: primary beam antennas + secondary beam antennas
            Nant = len(beam_rolls[0]) + len(sec_beam_rolls[0])

            for channel in station.iter_channels():  # loop over all channels (i.e. antennas) of the station
                channel_id = channel.get_id()
                if channel_id not in triggered_channels:  # skip all channels that do not participate in the trigger decision
                    logger.debug("skipping channel{}".format(channel_id))
                    continue

                trace = channel.get_trace()  # get the time trace (i.e. an array of amplitudes)
                times = channel.get_times()  # get the corresponding time bins

                if(phased_trace is None):
                    phased_trace = np.roll(trace, subbeam_rolls[channel_id])
                else:
                    phased_trace += np.roll(trace, subbeam_rolls[channel_id])

                if(channel_id in sec_subbeam_rolls):
                    #pass
                    phased_trace += np.roll(trace, sec_subbeam_rolls[channel_id])

            # Implmentation of the ARA-like power trigger
            window_width = int(window_time/time_step)
            n_windows = int(len(trace)/window_width)
            n_windows = 2*n_windows - 1
            squared_mean_threshold = Nant * threshold**2

            # Create a sliding window
            strides = phased_trace.strides
            windowed_traces = np.lib.stride_tricks.as_strided(phased_trace, \
                              shape=(n_windows,window_width), \
                              strides=(int(window_width/2)*strides[0], strides[0]))

            squared_mean = np.sum(windowed_traces**2/window_width, axis=1)

            if True in (squared_mean > squared_mean_threshold):
                logger.debug("Station has triggered")
                return True

        return False

    def run(self, evt, station, det,
            threshold=60 * units.mV,
            triggered_channels=None,
            secondary_channels=None,
            trigger_name='simple_phased_threshold',
            phasing_angles=default_angles,
            secondary_phasing_angles=default_sec_angles,
            set_not_triggered=False,
            window_time=10.67*units.ns,
            only_primary=False,
            coupled=True,
            ref_index=1.55):
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
        set_not_triggered: bool (default: False)
            if True not trigger simulation will be performed and this trigger will be set to not_triggered
        window_time: float
            Width of the time window used in the power integration
        only_primary: bool
            if True, no secondary beams are formed
        coupled: bool
            if True, the primary sub-beams are paired to the secondary sub-beams.
            if False, the primary sub-beams and the secondary sub-beams specify independent beams.
        ref_index: float
            refractive index for beam forming

        Returns
        -------
        is_triggered: bool
            True if the triggering condition is met
        """

        if (triggered_channels == None):
        	triggered_channels = [channel._id for channel in station.iter_channels()]

        if (secondary_channels == None):
            # If there are no chosen secondary channels, we take two consecutive
            # antennas, discard the third, and so on.
            secondary_channels = [ channel._id for channel in station.iter_channels() \
                                   if not channel._id % 3 == 2 ]

        if set_not_triggered:

            is_triggered = False

        else:

            logger.debug("primary channels:", triggered_channels)
            beam_rolls = self.get_beam_rolls(station, det, triggered_channels, phasing_angles, ref_index=ref_index)
            logger.debug("secondary_channels:", secondary_channels)
            secondary_beam_rolls = self.get_beam_rolls(station, det, secondary_channels, secondary_phasing_angles, ref_index=ref_index)

            if only_primary:
                empty_rolls = [ {} for direction in range(len(phasing_angles)) ]
                is_triggered = self.phased_trigger(station, beam_rolls, empty_rolls, triggered_channels, threshold, window_time)
            elif coupled:
                is_triggered = self.phased_trigger(station, beam_rolls, secondary_beam_rolls, triggered_channels, threshold, window_time)
            else:
                empty_rolls = [ {} for direction in range(len(phasing_angles)) ]
                primary_trigger = self.phased_trigger(station, beam_rolls, empty_rolls, triggered_channels, threshold, window_time)
                secondary_trigger = self.phased_trigger(station, secondary_beam_rolls, empty_rolls, secondary_channels, threshold, window_time)
                is_triggered = primary_trigger or secondary_trigger

        trigger = SimplePhasedTrigger(trigger_name, threshold, triggered_channels, secondary_channels, phasing_angles, secondary_phasing_angles)
        trigger.set_triggered(is_triggered)
        station.set_trigger(trigger)

        return is_triggered

    def end(self):
        pass
