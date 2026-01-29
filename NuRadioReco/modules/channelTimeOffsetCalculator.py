import numpy as np
import radiotools.helper
from NuRadioReco.utilities import fft
import NuRadioReco.framework.base_trace
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.detector.antennapattern
import NuRadioMC.SignalProp.analyticraytracing
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage


class channelTimeOffsetCalculator:
    """
    Determines the difference in signal arrival times between channels, the ray types and the signal
    arrival direction from the vertex position.

    The module uses either the vertex position stored in one of the sim showers or it assumes that
    some vertex reconstruction module was already run and stored the vertex position either in the
    vertex_2D_fit (for the neutrino2DVertexReconstructor module) of the nu_vertex station parameter.
    
    The module then calculates the expected difference in travel times between channels for this
    vertex position, shifts the channel voltages by that time difference and calculates the correlation
    with a template. By adding up the correlations for all channels, we can determine the correct
    raytracing solution: If the solution is correctd, the correlations will have their maxima at the same
    position, resulting in a larger maximum of their sum. Thus, we can determine the correct raytracing
    solutions and store the corresponding properties in the channel parameters.
    
    This module assumes that the ray path type for all channels is the same, e.g. each channel sees a direct
    ray. Therefore it should only be used for channels that are relatively close to each other.
    
    """
    def __init__(self):
        self.__use_sim = False
        self.__raytracing_types = ['direct', 'refracted', 'reflected']
        self.__electric_field_template = None
        self.__medium = None
        self.__antenna_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()

    def begin(self, electric_field_template, medium, use_sim=False):
        """
        Setup method for the module

        Parameters
        ----------
        electric_field_template: NuRadioReco.framework.base_trace.BaseTrace object (or child of BaseTrace class)
            An electric field waveform to be used as a template. It will be folded with the antenna and amplifier
            response to get a voltage template, which is then used to estimate timing differences between channels.
        medium: NuRadioMC.utilities.medium.Medium object
            The index of refraction profile of the ice.
        use_sim: Boolean (default: False)
            If true, the simulated vertex is used, otherwise it is assumed that the vertex is stored on either the
            vertex_2D_fit (for the neutrino2DVertexReconstructor) or the nu_vertex station parameter
        """
        self.__use_sim = use_sim
        self.__electric_field_template = electric_field_template
        self.__medium = medium

    @register_run()
    def run(self, event, station, det, channel_ids, passband):
        """
        Run the module on a station

        Parameters
        ----------
        event: NuRadioReco.framework.event.Event object
            The event the module should be run on
        station: NuRadioReco.framework.station.Station object
            The station the module should be run on
        det: NuRadioReco.detector.detector.Detector object of child object
            The detector description
        channel_ids: array of int
            IDs of the channels the module should be run on
        passband: array of float
            Lower and upper bound of the bandpass filter that is applied to the
            channel waveforms when determining correlation to the template. Can
            be used to filter out frequencies that are dominated by noise.
        """

        # Create data structured to store pulse properties
        propagation_times = np.zeros((len(channel_ids), 3))
        receive_angles = np.zeros((len(channel_ids), 3, 2))
        found_solutions = np.zeros((len(channel_ids), 3), dtype=bool)
        # Get vertex position
        vertex_position = None
        use_2d_vertex = False
        if self.__use_sim:
            for sim_shower in event.get_sim_showers():
                if sim_shower.has_parameter(shp.vertex):
                    vertex_position = sim_shower.get_parameter(shp.vertex)
                    use_2d_vertex = False
                    break
        else:
            if station.has_parameter(stnp.nu_vertex):
                vertex_position = station.get_parameter(stnp.nu_vertex)
                use_2d_vertex = False
            elif station.has_parameter(stnp.vertex_2D_fit):
                vertex_2d = station.get_parameter(stnp.vertex_2D_fit)
                vertex_position = np.array([vertex_2d[0], 0, vertex_2d[1]])
                use_2d_vertex = True
        if vertex_position is None:
            raise RuntimeError('Could not find vertex position')
        channel_time_ranges = np.zeros((len(channel_ids), 2))
        raytracer = NuRadioMC.SignalProp.analyticraytracing.ray_tracing(self.__medium)
        for i_channel, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            channel_time_ranges[i_channel, 0] = channel.get_trace_start_time()
            channel_time_ranges[i_channel, 1] = channel.get_times()[-1]
            raytracer.set_start_and_end_point(vertex_position, det.get_relative_position(station.get_id(), channel_id))
            raytracer.find_solutions()
            for i_solution, solution in enumerate(raytracer.get_results()):
                propagation_times[i_channel, solution['type'] - 1] = raytracer.get_travel_time(i_solution)
                found_solutions[i_channel, solution['type'] - 1] = True
                receive_verctor = raytracer.get_receive_vector(i_solution)
                receive_angles[i_channel, solution['type'] - 1] = radiotools.helper.cartesian_to_spherical(*receive_verctor)
        sampling_rate = station.get_channel(channel_ids[0]).get_sampling_rate()
        if self.__electric_field_template.get_sampling_rate() != sampling_rate:
            self.__electric_field_template.resample(sampling_rate)
        n_samples = int(np.round((np.max(channel_time_ranges) - np.min(channel_time_ranges))) * sampling_rate)
        n_samples += n_samples % 2
        channel_templates = np.zeros((len(channel_ids), 4, n_samples))
        correlation_sum = np.zeros(n_samples * 2 - 1)
        time_offsets = np.arange(-len(correlation_sum) // 2, len(correlation_sum) // 2) / sampling_rate
        propagation_times -= np.min(propagation_times[found_solutions])
        empty_trace = NuRadioReco.framework.base_trace.BaseTrace()
        empty_trace.set_trace(np.zeros(n_samples), sampling_rate)
        empty_trace.set_trace_start_time(np.min(channel_time_ranges))
        plt.close('all')
        for i_channel, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            for i_solution in range(3):
                if found_solutions[i_channel, i_solution]:
                    amp_response = det.get_amplifier_response(station.get_id(), channel_id,
                                                              self.__electric_field_template.get_frequencies())
                    antenna_orientation = det.get_antenna_orientation(station.get_id(), channel_id)
                    antenna_pattern = self.__antenna_provider.load_antenna_pattern(
                        det.get_antenna_model(station.get_id(), channel_id))
                    antenna_response = antenna_pattern.get_antenna_response_vectorized(
                        self.__electric_field_template.get_frequencies(),
                        receive_angles[i_channel, i_solution, 0],
                        receive_angles[i_channel, i_solution, 1],
                        antenna_orientation[0],
                        antenna_orientation[1],
                        antenna_orientation[2],
                        antenna_orientation[3]
                    )
                    channel_spectrum_template = fft.time2freq(
                        self.__electric_field_template.get_filtered_trace(passband, filter_type='butterabs'),
                        self.__electric_field_template.get_sampling_rate()
                    ) * amp_response * (antenna_response['theta'] + antenna_response['phi'])
                    channel_trace_template = fft.freq2time(channel_spectrum_template, sampling_rate)
                    start_bin = int(propagation_times[i_channel, i_solution] * sampling_rate)
                    channel_templates[i_channel, 0][start_bin:start_bin + len(channel_trace_template)] += channel_trace_template
                    channel_templates[i_channel, i_solution + 1][start_bin:start_bin + len(channel_trace_template)] = channel_trace_template

            if np.max(channel_templates[i_channel, 0]) > 0:
                correlation_sum += radiotools.helper.get_normalized_xcorr(
                    channel_templates[i_channel, 0],
                    (channel + empty_trace).get_filtered_trace(passband, 'butter', 10)
                )

        for i_channel, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            if channel.has_parameter(chp.signal_regions):
                signal_regions = channel.get_parameter(chp.signal_regions)
                if len(signal_regions) > 0:
                    time_offsets = np.zeros(len(signal_regions))
                    receiving_zeniths = np.zeros(len(signal_regions))
                    receiving_azimuths = np.zeros(len(signal_regions))
                    ray_types = np.zeros(len(signal_regions))

                    for i_region, signal_region in enumerate(signal_regions):
                        signal_window_trace = (channel + empty_trace).get_filtered_trace(passband, 'butter', 10)
                        signal_window_times = (channel + empty_trace).get_times()
                        signal_window_trace[signal_window_times < signal_region[0]] = 0
                        signal_window_trace[signal_window_times > signal_region[1]] = 0
                        max_correlations = np.zeros(3)
                        for i_ray in range(3):
                            if found_solutions[i_channel, i_ray]:
                                signal_window_correlation = radiotools.helper.get_normalized_xcorr(channel_templates[i_channel, i_ray + 1], signal_window_trace)
                                max_correlations[i_ray] = np.max(scipy.ndimage.maximum_filter(np.abs(signal_window_correlation), size=20) * scipy.ndimage.maximum_filter(np.abs(correlation_sum), size=20))
                        max_corr_ray = np.argmax(max_correlations)
                        time_offsets[i_region] = propagation_times[i_channel, max_corr_ray]
                        receiving_zeniths[i_region] = receive_angles[i_channel, max_corr_ray, 0]
                        receiving_azimuths[i_region] = receive_angles[i_channel, max_corr_ray, 1]
                        ray_types[i_region] = max_corr_ray + 1
                    channel.set_parameter(chp.signal_time_offsets, time_offsets)
                    channel.set_parameter(chp.signal_receiving_zeniths, receiving_zeniths)
                    if not use_2d_vertex:
                        channel.set_parameter(chp.signal_receiving_azimuths, receiving_azimuths)
                        channel.set_parameter(chp.signal_ray_types, ray_types)
                else:
                    channel.set_parameter(chp.signal_ray_types, [])
                    channel.set_parameter(chp.signal_receiving_zeniths, [])
                    channel.set_parameter(chp.signal_receiving_azimuths, [])

