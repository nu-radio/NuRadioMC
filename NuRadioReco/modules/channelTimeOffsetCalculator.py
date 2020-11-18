import numpy as np
import radiotools.helper
from NuRadioReco.utilities import fft
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.detector.antennapattern
import NuRadioMC.SignalProp.analyticraytracing
import matplotlib.pyplot as plt


class channelTimeOffsetCalculator:

    def __init__(self):
        self.__use_sim = False
        self.__raytracing_types = ['direc', 'refracted', 'reflected']
        self.__electric_field_template = None
        self.__medium = None
        self.__antenna_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
        self.__channel_trace_templates = None
        self.__debug = False

    def begin(self, electric_field_template, medium, use_sim=False, debug=False):
        self.__use_sim = use_sim
        self.__electric_field_template = electric_field_template
        self.__medium = medium
        self.__debug = debug

    @register_run()
    def run(self, event, station, det, channel_ids, passband):
        propagation_times = np.zeros((len(channel_ids), 3))
        receive_angles = np.zeros((len(channel_ids), 3))
        found_solutions = np.zeros((len(channel_ids), 3))
        self.__channel_trace_templates = np.zeros((len(channel_ids), 3, len(self.__electric_field_template.get_trace())))
        vertex_position = None
        if self.__use_sim:
            for sim_shower in event.get_sim_showers():
                if sim_shower.has_parameter(shp.vertex):
                    vertex_position = sim_shower.get_parameter(shp.vertex)
                    break
        else:
            if station.has_parameter(stnp.vertex_2D_fit):
                vertex_2d = station.get_parameter(stnp.vertex_2D_fit)
                vertex_position = np.array([vertex_2d[0], 0, vertex_2d[1]])
        if vertex_position is None:
            raise ValueError('Could not find vertex position')
        correlation_size = 0
        for i_channel, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            if channel.get_number_of_samples() + self.__electric_field_template.get_number_of_samples() + 1 > correlation_size:
                correlation_size = channel.get_number_of_samples() + self.__electric_field_template.get_number_of_samples() + 1
            channel_position = det.get_relative_position(101, channel_id)
            raytracer = NuRadioMC.SignalProp.analyticraytracing.ray_tracing(
                x1=vertex_position,
                x2=channel_position,
                medium=self.__medium
            )
            raytracer.find_solutions()
            for i_solution, solution in enumerate(raytracer.get_results()):
                solution_type = raytracer.get_solution_type(i_solution)
                found_solutions[i_channel, solution_type - 1] += 1
                propagation_times[i_channel, solution_type - 1] = raytracer.get_travel_time(i_solution)
                receive_vector = raytracer.get_receive_vector(i_solution)
                receive_angles[i_channel, solution_type - 1] = radiotools.helper.cartesian_to_spherical(receive_vector[0], receive_vector[1], receive_vector[2])[0]
        for i_solution in range(3):
            if len(propagation_times[:, i_solution][propagation_times[:, i_solution] > 0]) > 0:
                propagation_times[:, i_solution][propagation_times[:, i_solution] > 0] -= np.mean(propagation_times[:, i_solution][propagation_times[:, i_solution] > 0])
        correlation_sum = np.zeros((3, correlation_size))
        max_channel_length = 0
        for i_channel, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            antenna_pattern = self.__antenna_provider.load_antenna_pattern(det.get_antenna_model(101, channel_id))
            antenna_orientation = det.get_antenna_orientation(101, channel_id)
            if channel.get_number_of_samples() > max_channel_length:
                max_channel_length = channel.get_number_of_samples()
            for i_solution in range(3):
                if found_solutions[i_channel, i_solution] > 0:
                    antenna_response = antenna_pattern.get_antenna_response_vectorized(
                        self.__electric_field_template.get_frequencies(),
                        receive_angles[i_channel, i_solution],
                        0.,
                        antenna_orientation[0],
                        antenna_orientation[1],
                        antenna_orientation[2],
                        antenna_orientation[3]
                    )
                    channel_template_spec = fft.time2freq(self.__electric_field_template.get_filtered_trace(passband), self.__electric_field_template.get_sampling_rate()) * \
                        det.get_amplifier_response(101, channel_id, self.__electric_field_template.get_frequencies()) * (antenna_response['theta'] + antenna_response['phi'])
                    channel_template_trace = fft.freq2time(channel_template_spec, self.__electric_field_template.get_sampling_rate())
                    self.__channel_trace_templates[i_channel, i_solution] = channel_template_trace
                    channel.apply_time_shift(-propagation_times[i_channel, i_solution], True)
                    correlation = radiotools.helper.get_normalized_xcorr(channel_template_trace, channel.get_filtered_trace(passband))
                    correlation = np.abs(correlation)
                    correlation_sum[i_solution][:len(correlation)] += correlation
                    channel.apply_time_shift(propagation_times[i_channel, i_solution], True)

        correlation_sum_max = np.max(correlation_sum, axis=1)
        correlation_sum = np.zeros(self.__electric_field_template.get_number_of_samples() + max_channel_length)
        fig, ax = plt.subplots(1, 2)
        ax[0].grid()
        ax[1].grid()
        for i_channel, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            i_max = np.argmax(correlation_sum_max)
            channel.set_parameter(chp.signal_time_offset, propagation_times[i_channel, i_max])
            channel.set_parameter(chp.signal_receiving_zenith, receive_angles[i_channel, i_max])
            channel.set_parameter(chp.signal_ray_type, self.__raytracing_types[i_max])
            channel.apply_time_shift(-channel.get_parameter(chp.signal_time_offset))
            if np.max(self.__channel_trace_templates[i_channel, i_max]) > 0:
                corr = radiotools.helper.get_normalized_xcorr(self.__channel_trace_templates[i_channel, i_max], channel.get_trace())
                corr[np.isnan(corr)] = 0
                correlation_sum[:len(corr)] += corr
                ax[0].plot(channel.get_times() - channel.get_trace_start_time(), channel.get_trace() / np.max(channel.get_trace()))
                toffset = -(np.arange(0, correlation_sum.shape[0]) - max_channel_length) / channel.get_sampling_rate()
            channel.apply_time_shift(channel.get_parameter(chp.signal_time_offset))

        ax[1].plot(toffset, correlation_sum)
        ax[0].plot(
            np.arange(len(self.__channel_trace_templates[0, i_max])) / channel.get_sampling_rate() + toffset[np.argmax(correlation_sum)],
            self.__channel_trace_templates[0, i_max] / np.max(self.__channel_trace_templates[0, i_max]),
            c='k'
        )
        for i_channel, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            channel.set_parameter(chp.signal_time, toffset[np.argmax(correlation_sum)])
        # plt.show()
