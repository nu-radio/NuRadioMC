import numpy as np
import time
import logging

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import units, signal_processing
from NuRadioReco.modules.efieldToVoltageConverter import calculate_time_shift_for_cosmic_ray

import NuRadioReco.framework.sim_channel
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import channelParameters as chp


class efieldToVoltageConverterPerEfield():
    """
    This module applies the antenna response to each electric field individually and stores the
    resulting voltage traces in the SimStationclass as SimChannel objects
    """

    def __init__(self, log_level=logging.NOTSET):
        self.__t = 0
        self.logger = logging.getLogger('NuRadioReco.efieldToVoltageConverterPerEfield')
        self.logger.setLevel(log_level)
        self.antenna_provider = antennapattern.AntennaPatternProvider()

    @register_run()
    def run(self, evt, station, det):
        """
        Converts simulated electric fields to voltage traces for each channel in a sim station.

        Parameters
        ----------
        evt : Event
            The event object.
        station : SimStation (or Station)
            The SimStation object. If a Station object is provided, it is detected and the
            SimStation is automatically retrived from the Station object
        det : Detector
            The detector object.

        Returns
        -------
        None

        Raises
        ------
        LookupError
            If the station has no electric fields.
        """
        t = time.time()

        # access simulated efield and high level parameters
        if isinstance(station, NuRadioReco.framework.station.Station):
            sim_station = station.get_sim_station()
        else:
            sim_station = station
        if(len(sim_station.get_electric_fields()) == 0):
            raise LookupError(f"station {station.get_id()} has no efields")

        for channel_id in det.get_channel_ids(sim_station.get_id()):
            # one channel might contain multiple channels to store the signals from multiple ray paths and showers,
            # so we loop over all simulated channels with the same id,
            self.logger.debug('channel id {}'.format(channel_id))
            for electric_field in sim_station.get_electric_fields_for_channels([channel_id]):
                sim_channel = NuRadioReco.framework.sim_channel.SimChannel(channel_id, shower_id=electric_field.get_shower_id(),
                                                                           ray_tracing_id=electric_field.get_ray_tracing_solution_id())
                sim_channel[chp.signal_ray_type] = electric_field[efp.ray_path_type]

                ff = electric_field.get_frequencies()
                efield_fft = electric_field.get_frequency_spectrum()

                zenith = electric_field[efp.zenith]
                azimuth = electric_field[efp.azimuth]

                # get antenna pattern for current channel
                VEL = signal_processing.get_efield_antenna_factor(sim_station, ff, [channel_id], det, zenith, azimuth, self.antenna_provider)

                if VEL is None:  # this can happen if there is not signal path to the antenna
                    voltage_fft = np.zeros_like(efield_fft[1])  # set voltage trace to zeros
                else:
                    # Apply antenna response to electric field
                    VEL = VEL[0]  # we only requested the VEL for one channel, so selecting it
                    voltage_fft = np.sum(VEL * np.array([efield_fft[1], efield_fft[2]]), axis=0)

                # Remove DC offset
                voltage_fft[np.where(ff < 5 * units.MHz)] = 0.

                dist_channel_efield = np.linalg.norm(det.get_relative_position(sim_station.get_id(), channel_id) - electric_field.get_position())
                if dist_channel_efield / units.mm > 0.01:
                    travel_time_shift = calculate_time_shift_for_cosmic_ray(
                        det, sim_station, electric_field, channel_id)
                else:
                    travel_time_shift = 0

                # set the trace to zeros
                sim_channel.set_frequency_spectrum(voltage_fft, electric_field.get_sampling_rate())
                sim_channel.set_trace_start_time(electric_field.get_trace_start_time() + travel_time_shift)
                sim_station.add_channel(sim_channel)

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        return dt
