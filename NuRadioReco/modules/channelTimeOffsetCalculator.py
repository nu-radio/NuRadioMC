import numpy as np
import radiotools.helper
from NuRadioReco.utilities import fft
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.detector.antennapattern
import NuRadioMC.SignalProp.analyticraytracing


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
        receive_angles = np.zeros((len(channel_ids), 3))
        found_solutions = np.zeros((len(channel_ids), 3))
        # Get vertex position
        vertex_position = None
        if self.__use_sim:
            for sim_shower in event.get_sim_showers():
                if sim_shower.has_parameter(shp.vertex):
                    vertex_position = sim_shower.get_parameter(shp.vertex)
                    break
        else:
            if station.has_parameter(stnp.nu_vertex):
                vertex_position = station.get_parameter(stnp.nu_vertex)
            elif station.has_parameter(stnp.vertex_2D_fit):
                vertex_2d = station.get_parameter(stnp.vertex_2D_fit)
                vertex_position = np.array([vertex_2d[0], 0, vertex_2d[1]])
        if vertex_position is None:
            raise RuntimeError('Could not find vertex position')
        correlation_size = 0
        for i_channel, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            if channel.get_sampling_rate() != self.__electric_field_template.get_sampling_rate():
                raise RuntimeError(
                    'The channels and the electric field remplate need to have the same sampling rate.'
                )
            # Calculate size of largest autocorrelation
            if channel.get_number_of_samples() + self.__electric_field_template.get_number_of_samples() - 1 > correlation_size:
                correlation_size = channel.get_number_of_samples() + self.__electric_field_template.get_number_of_samples() - 1
            channel_position = det.get_relative_position(station.get_id(), channel_id)
            raytracer = NuRadioMC.SignalProp.analyticraytracing.ray_tracing(
                x1=vertex_position,
                x2=channel_position,
                medium=self.__medium
            )
            raytracer.find_solutions()
            # Loop through all 3 ray path types and store the properties into the data structure
            for i_solution, solution in enumerate(raytracer.get_results()):
                solution_type = raytracer.get_solution_type(i_solution)
                found_solutions[i_channel, solution_type - 1] += 1
                propagation_times[i_channel, solution_type - 1] = raytracer.get_travel_time(i_solution)
                receive_vector = raytracer.get_receive_vector(i_solution)
                receive_angles[i_channel, solution_type - 1] = radiotools.helper.cartesian_to_spherical(receive_vector[0], receive_vector[1], receive_vector[2])[0]
        # We only want the relative time differences between channels, so we subtract the mean propagation time
        for i_solution in range(3):
            if len(propagation_times[:, i_solution][propagation_times[:, i_solution] > 0]) > 0:
                propagation_times[:, i_solution][propagation_times[:, i_solution] > 0] -= np.mean(propagation_times[:, i_solution][propagation_times[:, i_solution] > 0])
        correlation_sum = np.zeros((3, correlation_size))
        # Now we check which ray path results in the best correlation between channels
        for i_channel, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            antenna_pattern = self.__antenna_provider.load_antenna_pattern(det.get_antenna_model(station.get_id(), channel_id))
            antenna_orientation = det.get_antenna_orientation(station.get_id(), channel_id)
            for i_solution in range(3):
                if found_solutions[i_channel, i_solution] > 0:
                    # We calculate the voltage template from the electric field template using the receiving angles
                    # for that raytracing solution
                    antenna_response = antenna_pattern.get_antenna_response_vectorized(
                        self.__electric_field_template.get_frequencies(),
                        receive_angles[i_channel, i_solution],
                        0.,
                        antenna_orientation[0],
                        antenna_orientation[1],
                        antenna_orientation[2],
                        antenna_orientation[3]
                    )
                    # For simplicity, we assume equal contribution on the E_theta and E_phi component
                    channel_template_spec = fft.time2freq(self.__electric_field_template.get_filtered_trace(passband), self.__electric_field_template.get_sampling_rate()) * \
                        det.get_amplifier_response(station.get_id(), channel_id, self.__electric_field_template.get_frequencies()) * (antenna_response['theta'] + antenna_response['phi'])
                    channel_template_trace = fft.freq2time(channel_template_spec, self.__electric_field_template.get_sampling_rate())
                    # We apply the expected time shift for the raytracing solution and calculate the correlation with the template
                    channel.apply_time_shift(-propagation_times[i_channel, i_solution], True)
                    correlation = radiotools.helper.get_normalized_xcorr(channel_template_trace, channel.get_filtered_trace(passband))
                    correlation = np.abs(correlation)
                    correlation_sum[i_solution][:len(correlation)] += correlation
                    channel.apply_time_shift(propagation_times[i_channel, i_solution], True)

        correlation_sum_max = np.max(correlation_sum, axis=1)
        # We look for the raytracing solution that yielded the best correlation and store the corresponding properties
        # in the channel parameters
        for i_channel, channel_id in enumerate(channel_ids):
            channel = station.get_channel(channel_id)
            channel.set_parameter(chp.signal_time_offset, propagation_times[i_channel, np.argmax(correlation_sum_max)])
            channel.set_parameter(chp.signal_receiving_zenith, receive_angles[i_channel, np.argmax(correlation_sum_max)])
            channel.set_parameter(chp.signal_ray_type, self.__raytracing_types[np.argmax(correlation_sum_max)])
