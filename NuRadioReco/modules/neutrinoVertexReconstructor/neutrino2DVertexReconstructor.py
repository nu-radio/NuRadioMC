import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, fft
import NuRadioReco.utilities.io_utilities
import NuRadioReco.framework.electric_field
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
import radiotools.helper as hp
import NuRadioMC.SignalProp.analyticraytracing
import NuRadioMC.utilities.medium
import NuRadioMC.SignalGen.askaryan


class neutrino2DVertexReconstructor:

    def __init__(self, lookup_table_location):
        """
        Constructor for the vertex reconstructor

        Parameters
        --------------
        lookup_table_location: string
            path to the folder in which the lookup tables for the signal travel
            times are stored
        """
        self.__lookup_table_location = lookup_table_location
        self.__detector = None
        self.__lookup_table = {}
        self.__header = {}
        self.__channel_ids = None
        self.__station_id = None
        self.__channel_pairs = None
        self.__rec_x = None
        self.__rec_z = None
        self.__sampling_rate = None
        self.__channel_pair = None
        self.__channel_positions = None
        self.__correlation = None
        self.__max_corr_index = None
        self.__current_ray_types = None
        self.__passband = None
        self.__template = None
        self.__output_path = None
        self.__ray_types = [
            ['direct', 'direct'],
            ['reflected', 'reflected'],
            ['refracted', 'refracted'],
            ['direct', 'reflected'],
            ['reflected', 'direct'],
            ['direct', 'refracted'],
            ['refracted', 'direct'],
            ['reflected', 'refracted'],
            ['refracted', 'reflected']
        ]
        self.__dnr_ray_types = [
            ['direct', 'reflected'],
            ['reflected', 'direct'],
            ['direct', 'refracted'],
            ['refracted', 'direct'],
            ['reflected', 'refracted'],
            ['refracted', 'reflected']
        ]

    def begin(self, station_id, channel_ids, detector, passband=None, template=None, output_path=None):
        """
        General settings for vertex reconstruction

        Parameters
        -------------
        station_id: integer
            ID of the station to be used for the reconstruction
        channel_ids: array of integers
            IDs of the channels to be used for the reconstruction
        detector: Detector or GenericDetector
            Detector description for the detector used in the reconstruction
        passband: array of float or None
            Passband of the filter that should be applied to channel traces before
            calculating the correlation. If None is passed, no filter is applied
        template: array of float or none
            Waveform template with which the channel waveforms are correlated to determine
            the timing differences. If None is passed, the channels are correlated directly
            with each other
        output_path: string or None
            Location where plots of the reconstruction are saved. If None is passed, no plots are
            created.
        """
        first_channel_position = detector.get_relative_position(station_id, channel_ids[0])
        for channel_id in channel_ids:
            pos = detector.get_relative_position(station_id, channel_id)
            # Check if channels are on the same string. Allow some tolerance for
            # uncertainties from deployment
            if np.abs(pos[0] - first_channel_position[0]) > 1. * units.m or np.abs(pos[1] - first_channel_position[1]) > 1. * units.m:
                raise ValueError('All channels have to be on the same string')
        self.__detector = detector
        self.__channel_ids = channel_ids
        self.__station_id = station_id
        self.__channel_pairs = []
        for i in range(len(channel_ids) - 1):
            for j in range(i + 1, len(channel_ids)):
                self.__channel_pairs.append([channel_ids[i], channel_ids[j]])
        self.__lookup_table = {}
        self.__header = {}
        self.__passband = passband
        for channel_id in channel_ids:
            channel_z = abs(detector.get_relative_position(station_id, channel_id)[2])
            if channel_z not in self.__lookup_table.keys():
                f = NuRadioReco.utilities.io_utilities.read_pickle('{}/lookup_table_{}.p'.format(self.__lookup_table_location, int(abs(channel_z))))
                self.__header[int(channel_z)] = f['header']
                self.__lookup_table[int(abs(channel_z))] = f['antenna_{}'.format(channel_z)]
        self.__template = template
        self.__output_path = output_path

    def run(self, event, station, max_distance, z_width, grid_spacing, direction_guess=None, debug=False, use_dnr=False):
        """
        Execute the 2D vertex reconstruction

        Parameters
        ---------------
        station: Station
            The station for which the vertex shall be reconstructed
        max_distance: number
            Maximum distance up to which the vertex position shall be searched
        z_width: number
            Vertical size of the search area. If direction_guess is specified, a
            stripe of z_width to each side of the initial direction will be searched.
            If direction_guess is not specified, z_width is the maximum depth up
            to which the vertex will be searched.
        grid_spacing: number
            Distance between two points of the grid on which the vertex is searched
        direction_guess: number, defaults to None
            Zenith for an initial guess of the vertex direction. If specified,
            a strip if width 2*z_width around the guessed direction will be searched
        debug: boolean
            If True, debug plots will be produced
        use_dnr: boolean
            If True, DnR pulses are included in the reconstruction by correlating
            the channel waveforms with themselves.
        """
        distances = np.arange(50. * units.m, max_distance, grid_spacing)
        if direction_guess is None:
            heights = np.arange(-z_width, 0, grid_spacing)
        else:
            heights = np.arange(-z_width, z_width, grid_spacing)
        x_0, z_0 = np.meshgrid(distances, heights)
        # Create list of coordinates at which we look for the vertex position
        # If we have an initial guess for the vertex direction, we only check possible vertex locations around that
        # direction, otherwise we search the whole space
        if direction_guess is None:
            x_coords = x_0
            z_coords = z_0
        else:
            x_coords = np.cos(direction_guess - 90. * units.deg) * x_0 + np.sin(direction_guess - 90. * units.deg) * z_0
            z_coords = -np.sin(direction_guess - 90. * units.deg) * x_0 + np.cos(direction_guess - 90. * units.deg) * z_0

        correlation_sum = np.zeros(x_coords.shape)

        corr_range = 50. * units.ns
        for i_pair, channel_pair in enumerate(self.__channel_pairs):
            ch1 = station.get_channel(channel_pair[0])
            ch2 = station.get_channel(channel_pair[1])
            snr1 = np.max(np.abs(ch1.get_trace()))
            snr2 = np.max(np.abs(ch2.get_trace()))
            if snr1 == 0 or snr2 == 0:
                continue
            spec1 = np.copy(ch1.get_frequency_spectrum())
            spec2 = np.copy(ch2.get_frequency_spectrum())
            if self.__passband is not None:
                b, a = scipy.signal.butter(10, self.__passband, 'bandpass', analog=True)
                w, h = scipy.signal.freqs(b, a, ch1.get_frequencies())
                spec1 *= h
                spec2 *= h
            trace1 = fft.freq2time(spec1, ch1.get_sampling_rate())
            trace2 = fft.freq2time(spec2, ch2.get_sampling_rate())
            if self.__template is not None:
                corr_1 = hp.get_normalized_xcorr(trace1, self.__template)
                corr_2 = hp.get_normalized_xcorr(trace2, self.__template)
                self.__correlation = np.zeros_like(corr_1)
                sample_shifts = np.arange(-len(corr_1) // 2, len(corr_1) // 2, dtype=int)
                toffset = sample_shifts / ch1.get_sampling_rate()
                for i_shift, shift_sample in enumerate(sample_shifts):
                    self.__correlation[i_shift] = np.max(corr_1 * np.roll(corr_2, shift_sample))
            else:
                t_max1 = ch1.get_times()[np.argmax(np.abs(trace1))]
                t_max2 = ch2.get_times()[np.argmax(np.abs(trace2))]
                if snr1 > snr2:
                    trace1[np.abs(ch1.get_times() - t_max1) > corr_range] = 0
                else:
                    trace2[np.abs(ch2.get_times() - t_max2) > corr_range] = 0
                self.__correlation = np.abs(scipy.signal.correlate(trace1, trace2))
                toffset = -(np.arange(0, self.__correlation.shape[0]) - self.__correlation.shape[0] / 2.) / ch1.get_sampling_rate()
                if np.sum(np.abs(self.__correlation)) > 0:
                    self.__correlation /= np.sum(np.abs(self.__correlation))
            corr_snr = np.max(self.__correlation) / np.mean(self.__correlation[self.__correlation > 0])
            self.__sampling_rate = ch1.get_sampling_rate()
            self.__channel_pair = channel_pair
            self.__channel_positions = [self.__detector.get_relative_position(self.__station_id, channel_pair[0]), self.__detector.get_relative_position(self.__station_id, channel_pair[1])]
            correlation_array = np.zeros_like(correlation_sum)
            # Check every hypothesis for which ray types the antennas might have detected
            for i_ray in range(len(self.__ray_types)):
                self.__current_ray_types = self.__ray_types[i_ray]
                correlation_array = np.maximum(self.get_correlation_array_2d(x_coords, z_coords), correlation_array)
            if np.max(correlation_array) > 0:
                if self.__template is None:
                    correlation_sum += correlation_array / np.max(correlation_array) * corr_snr
                else:
                    correlation_sum += correlation_array
            max_corr_index = np.unravel_index(np.argmax(correlation_sum), correlation_sum.shape)
            max_corr_r = x_coords[max_corr_index[0]][max_corr_index[1]]
            max_corr_z = z_coords[max_corr_index[0]][max_corr_index[1]]

            if debug:
                fig1 = plt.figure(figsize=(12, 4))
                fig2 = plt.figure(figsize=(8, 12))
                ax1_1 = fig1.add_subplot(1, 3, 1)
                ax1_2 = fig1.add_subplot(1, 3, 2, sharey=ax1_1)
                ax1_3 = fig1.add_subplot(1, 3, 3)
                ax1_1.plot(ch1.get_times(), ch1.get_trace() / units.mV, c='C0', alpha=.3)
                ax1_2.plot(ch2.get_times(), ch2.get_trace() / units.mV, c='C1', alpha=.3)
                ax1_1.plot(ch1.get_times()[np.abs(trace1) > 0], trace1[np.abs(trace1) > 0] / units.mV, c='C0', alpha=1)
                ax1_2.plot(ch2.get_times()[np.abs(trace2) > 0], trace2[np.abs(trace2) > 0] / units.mV, c='C1', alpha=1)
                ax1_1.plot(ch1.get_times()[:len(self.__template)], self.__template, c='k')
                ax1_1.set_xlabel('t [ns]')
                ax1_1.set_ylabel('U [mV]')
                ax1_1.set_title('Channel {}'.format(self.__channel_pair[0]))
                ax1_2.set_xlabel('t [ns]')
                ax1_2.set_ylabel('U [mV]')
                ax1_2.set_title('Channel {}'.format(self.__channel_pair[1]))

                ax1_3.plot(toffset, self.__correlation)
                ax1_3.set_title('$SNR_{corr}$=%.2f' % (corr_snr))
                ax1_1.grid()
                ax1_2.grid()
                ax1_3.grid()
                fig1.tight_layout()
                ax2_1 = fig2.add_subplot(211)
                ax2_2 = fig2.add_subplot(212)
                corr_plots = ax2_1.pcolor(x_coords, z_coords, correlation_array)
                sum_plots = ax2_2.pcolor(x_coords, z_coords, correlation_sum)
                fig2.colorbar(corr_plots, ax=ax2_1)
                fig2.colorbar(sum_plots, ax=ax2_2)
                sim_vertex = None
                for shower in event.get_sim_showers():
                    if shower.has_parameter(shp.vertex):
                        sim_vertex = shower.get_parameter(shp.vertex)
                if sim_vertex is not None:
                    ax2_1.axvline(np.sqrt(sim_vertex[0]**2 + sim_vertex[1]**2), c='r', linestyle=':')
                    ax2_1.axhline(sim_vertex[2], c='r', linestyle=':')
                    ax2_2.axvline(np.sqrt(sim_vertex[0]**2 + sim_vertex[1]**2), c='r', linestyle=':')
                    ax2_2.axhline(sim_vertex[2], c='r', linestyle=':')

                ax2_1.axvline(max_corr_r, c='k', linestyle=':')
                ax2_1.axhline(max_corr_z, c='k', linestyle=':')
                ax2_2.axvline(max_corr_r, c='k', linestyle=':')
                ax2_2.axhline(max_corr_z, c='k', linestyle=':')

                fig2.tight_layout()
                plt.show()
                plt.close('all')
        if use_dnr:
            dnr_correlation_sum = np.zeros(x_coords.shape)
            for channel_id in self.__channel_ids:
                channel = station.get_channel(channel_id)
                spec = channel.get_frequency_spectrum()
                if self.__passband is not None:
                    b, a = scipy.signal.butter(10, self.__passband, 'bandpass', analog=True)
                    w, h = scipy.signal.freqs(b, a, channel.get_frequencies())
                    spec *= h
                trace = fft.freq2time(spec, channel.get_sampling_rate())
                corr = hp.get_normalized_xcorr(trace, self.__template)
                self.__correlation = np.zeros_like(corr)
                sample_shifts = np.arange(-len(corr) // 2, len(corr) // 2, dtype=int)
                toffset = sample_shifts / channel.get_sampling_rate()
                for i_shift, shift_sample in enumerate(sample_shifts):
                    self.__correlation[i_shift] = np.max(corr * np.roll(corr, shift_sample))
                self.__correlation[np.abs(toffset) <= 5] = 0
                self.__sampling_rate = channel.get_sampling_rate()
                self.__channel_pair = [channel_id, channel_id]
                self.__channel_positions = [self.__detector.get_relative_position(self.__station_id, channel_id),
                                            self.__detector.get_relative_position(self.__station_id, channel_id)]
                correlation_array = np.zeros_like(correlation_sum)
                for i_ray in range(len(self.__dnr_ray_types)):
                    self.__current_ray_types = self.__dnr_ray_types[i_ray]
                    correlation_array = np.maximum(self.get_correlation_array_2d(x_coords, z_coords), correlation_array)
                if np.max(correlation_array) > 0:
                    dnr_correlation_sum += correlation_array
            max_corr_dnr_index = np.unravel_index(np.argmax(correlation_sum + dnr_correlation_sum), correlation_sum.shape)
            max_corr_dnr_r = x_coords[max_corr_dnr_index[0]][max_corr_dnr_index[1]]
            max_corr_dnr_z = z_coords[max_corr_dnr_index[0]][max_corr_dnr_index[1]]

        if self.__output_path is not None:
            plt.close('all')
            if use_dnr:
                fig3 = plt.figure(figsize=(12, 12))
                ax3_1 = fig3.add_subplot(321)
                ax3_2 = fig3.add_subplot(322)
                ax3_3 = fig3.add_subplot(3, 2, (3, 6))
            else:
                fig3 = plt.figure(figsize=(8, 8))
                ax3_1 = fig3.add_subplot(111)
            import skimage.transform
            downscaled_image = skimage.transform.rescale(correlation_sum, .2)
            rescaled_xcoords = skimage.transform.rescale(x_coords, .2)
            rescaled_zcoords = skimage.transform.rescale(z_coords, .2)
            corr_plot = ax3_1.pcolor(rescaled_xcoords, rescaled_zcoords, downscaled_image)
            ax3_1.grid()
            ax3_1.set_aspect('equal')
            plt.colorbar(corr_plot, ax=ax3_1)
            sim_vertex = None
            for shower in event.get_sim_showers():
                if shower.has_parameter(shp.vertex):
                    sim_vertex = shower.get_parameter(shp.vertex)
            if sim_vertex is not None:
                ax3_1.axvline(np.sqrt(sim_vertex[0] ** 2 + sim_vertex[1] ** 2), c='r', linestyle=':')
                ax3_1.axhline(sim_vertex[2], c='r', linestyle=':')
                ax3_1.axvline(max_corr_r, c='k', linestyle=':')
                ax3_1.axhline(max_corr_z, c='k', linestyle=':')
            if use_dnr:
                downscaled_dnr_image = skimage.transform.rescale(dnr_correlation_sum, .2)
                dnr_corr_plot = ax3_2.pcolor(rescaled_xcoords, rescaled_zcoords, downscaled_dnr_image)
                if np.max(downscaled_dnr_image) > .1:
                    ax3_2.contour(rescaled_xcoords, rescaled_zcoords, downscaled_dnr_image, levels=[.1], colors='k', alpha=.3)
                ax3_2.grid()
                ax3_2.set_aspect('equal')
                plt.colorbar(dnr_corr_plot, ax=ax3_2)
                combined_corr_plot = ax3_3.pcolor(rescaled_xcoords, rescaled_zcoords, downscaled_dnr_image + downscaled_image)
                ax3_3.grid()
                ax3_3.set_aspect('equal')
                plt.colorbar(combined_corr_plot, ax=ax3_3)
                if sim_vertex is not None:
                    ax3_2.axvline(np.sqrt(sim_vertex[0] ** 2 + sim_vertex[1] ** 2), c='r', linestyle=':')
                    ax3_2.axhline(sim_vertex[2], c='r', linestyle=':')
                    ax3_3.axvline(np.sqrt(sim_vertex[0] ** 2 + sim_vertex[1] ** 2), c='r', linestyle=':')
                    ax3_3.axhline(sim_vertex[2], c='r', linestyle=':')
                    ax3_3.axvline(max_corr_dnr_r, c='k', linestyle=':')
                    ax3_3.axhline(max_corr_dnr_z, c='k', linestyle=':')
            fig3.tight_layout()
            fig3.savefig('{}/vertex_reco_{}.png'.format(self.__output_path, event.get_id()))

        if max_corr_index is None:
            return
        self.__rec_x = x_coords[max_corr_index[0]][max_corr_index[1]]
        self.__rec_z = z_coords[max_corr_index[0]][max_corr_index[1]]
        station.set_parameter(stnp.vertex_2D_fit, [self.__rec_x, self.__rec_z])

        return

    def get_correlation_array_2d(self, x, z):
        """
        Returns the correlations corresponding to the different
        signal travel times between channels for the given positions.
        This is done by correcting for the distance of the channels
        from the station center and then calling
        self.get_correlation_for_pos, which does the actual work.

        Parameters
        ----------
        x, z: array
            Coordinates of the points for which calculations are
            to be calculated. Correspond to the (r, z) pair
            of cylindrical coordinates.
        """
        channel_pos1 = self.__channel_positions[0]
        channel_pos2 = self.__channel_positions[1]
        d_hor1 = np.sqrt((x - channel_pos1[0])**2 + (channel_pos1[1])**2)
        d_hor2 = np.sqrt((x - channel_pos2[0])**2 + (channel_pos2[1])**2)
        res = self.get_correlation_for_pos(np.array([d_hor1, d_hor2]), z)
        return res

    def get_correlation_for_pos(self, d_hor, z):
        """
        Returns the correlations corresponding to the different
        signal travel times between channels for the given positions.

        Parameters
        ----------
        d_hor, z: array
            Coordinates of the points for which calculations are
            to be calculated. Correspond to the (r, z) pair
            of cylindrical coordinates.
        """
        t1 = self.get_signal_travel_time(d_hor[0], z, self.__current_ray_types[0], self.__channel_pair[0])
        t2 = self.get_signal_travel_time(d_hor[1], z, self.__current_ray_types[1], self.__channel_pair[1])
        delta_t = t1 - t2
        delta_t = delta_t.astype(float)
        corr_index = self.__correlation.shape[0] / 2 + np.round(delta_t * self.__sampling_rate)
        corr_index[np.isnan(corr_index)] = 0
        mask = (~np.isnan(delta_t)) & (corr_index > 0) & (corr_index < self.__correlation.shape[0]) & (~np.isinf(delta_t))
        corr_index[~mask] = 0
        res = np.take(self.__correlation, corr_index.astype(int))
        res[~mask] = 0
        return res

    def get_signal_travel_time(self, d_hor, z, ray_type, channel_id):
        """
        Calculate the signal travel time between a position and the
        channel

        Parameters
        ----------
        d_hor, z: numbers or arrays of numbers
            Coordinates of the point from which to calculate the
            signal travel times. Correspond to (r, z) coordinates
            in cylindrical coordinates.
        ray_type: string
            Ray type for which to calculate the travel times. Options
            are direct, reflected and refracted
        channel_id: int
            ID of the channel to which the travel time shall be calculated
        """
        channel_pos = self.__detector.get_relative_position(self.__station_id, channel_id)
        channel_type = int(abs(channel_pos[2]))
        travel_times = np.zeros_like(d_hor)
        mask = np.ones_like(travel_times).astype(bool)
        channel_z = abs(self.__detector.get_relative_position(self.__station_id, channel_id)[2])
        if channel_z not in self.__lookup_table.keys():
            f = NuRadioReco.utilities.io_utilities.read_pickle(
                '{}/lookup_table_{}.p'.format(self.__lookup_table_location, int(abs(channel_z))))
            self.__header[int(channel_z)] = f['header']
            self.__lookup_table[int(abs(channel_z))] = f['antenna_{}'.format(channel_z)]

        i_x = np.array(np.round((-d_hor - self.__header[channel_type]['x_min']) / self.__header[channel_type]['d_x'])).astype(int)
        mask[i_x > self.__lookup_table[channel_type][ray_type].shape[0] - 1] = False
        i_z = np.array(np.round((z - self.__header[channel_type]['z_min']) / self.__header[channel_type]['d_z'])).astype(int)
        mask[i_z > self.__lookup_table[channel_type][ray_type].shape[1] - 1] = False
        i_x[~mask] = 0
        i_z[~mask] = 0
        travel_times = self.__lookup_table[channel_type][ray_type][i_x, i_z]
        travel_times[~mask] = np.nan
        return travel_times

    def find_ray_type(self, station, ch1):
        """
        Calculate the most likely ray type (direct, reflected
        or refracted) of the signal that reached the detector.
        This is done by taking the reconstructed vertex position,
        calculating the expected time offset between channels and
        checking for which ray type scenario the correlation
        between channels is largest.

        Parameters
        ----------
        station: station object
            The station on which this reconstruction was run
        ch1: channel object
            The channel for which the ray type shall be determined
        """
        corr_range = 50. * units.ns
        ray_types = ['direct', 'refracted', 'reflected']
        ray_type_correlations = np.zeros(3)
        for i_ray_type, ray_type in enumerate(ray_types):
            for channel_id in self.__channel_ids:
                if channel_id != ch1.get_id():
                    ch2 = station.get_channel(channel_id)
                    snr1 = np.max(np.abs(ch1.get_trace()))
                    snr2 = np.max(np.abs(ch2.get_trace()))
                    trace1 = np.copy(ch1.get_trace())
                    t_max1 = ch1.get_times()[np.argmax(np.abs(trace1))]
                    trace2 = np.copy(ch2.get_trace())
                    t_max2 = ch2.get_times()[np.argmax(np.abs(trace2))]
                    if snr1 > snr2:
                        trace1[np.abs(ch1.get_times() - t_max1) > corr_range] = 0
                    else:
                        trace2[np.abs(ch2.get_times() - t_max2) > corr_range] = 0
                    correlation = np.abs(scipy.signal.hilbert(scipy.signal.correlate(trace1, trace2)))
                    correlation /= np.sum(np.abs(correlation))
                    t_1 = self.get_signal_travel_time(np.array([self.__rec_x]), np.array([self.__rec_z]), ray_type, ch1.get_id())[0]
                    t_2 = self.get_signal_travel_time(np.array([self.__rec_x]), np.array([self.__rec_z]), ray_type, ch2.get_id())[0]
                    if np.isnan(t_1) or np.isnan(t_2):
                        return None
                    delta_t = t_1 - t_2
                    corr_index = correlation.shape[0] / 2 + np.round(delta_t * self.__sampling_rate)
                    if np.isnan(corr_index):
                        return None
                    corr_index = int(corr_index)
                    if 0 < corr_index < len(correlation):
                        ray_type_correlations[i_ray_type] += correlation[int(corr_index)]
        return ray_types[np.argmax(ray_type_correlations)]

    def find_receiving_zenith(self, station, ray_type, channel_id):
        solution_types = {1: 'direct',
                          2: 'refracted',
                          3: 'reflected'}
        nu_vertex_2D = station.get_parameter(stnp.vertex_2D_fit)
        nu_vertex = [nu_vertex_2D[0], 0, nu_vertex_2D[1]]
        ray_tracer = NuRadioMC.SignalProp.analyticraytracing.ray_tracing(
            nu_vertex,
            self.__detector.get_relative_position(station.get_id(), channel_id) + self.__detector.get_absolute_position(station.get_id()),
            NuRadioMC.utilities.medium.get_ice_model('greenland_simple')
        )
        ray_tracer.find_solutions()
        for i_solution, solution in enumerate(ray_tracer.get_results()):
            if solution_types[ray_tracer.get_solution_type(i_solution)] == ray_type:
                receive_vector = ray_tracer.get_receive_vector(i_solution)
                receive_zenith = hp.cartesian_to_spherical(receive_vector[0], receive_vector[1], receive_vector[2])[0]
                travel_time = ray_tracer.get_travel_time(i_solution)
                return receive_zenith, travel_time
        return None, None
