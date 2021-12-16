import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import NuRadioReco.utilities.io_utilities
import NuRadioReco.framework.electric_field
import NuRadioReco.detector.antennapattern
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.utilities import trace_utilities, fft, bandpass_filter
import radiotools.helper as hp


class neutrino3DVertexReconstructor:

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
        self.__passband = None
        self.__channel_pair = None
        self.__channel_positions = None
        self.__correlation = None
        self.__max_corr_index = None
        self.__current_ray_types = None
        self.__electric_field_template = None
        self.__azimuths_2d = None
        self.__distances_2d = None
        self.__z_coordinates_2d = None
        self.__distance_step_3d = None
        self.__z_step_3d = None
        self.__widths_3d = None
        self.__heights_3d = None
        self.__debug_folder = None
        self.__current_distance = None
        self.__pair_correlations = None
        self.__self_correlations = None
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
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

    def begin(
            self,
            station_id,
            channel_ids,
            detector,
            template,
            distances_2d=None,
            azimuths_2d=None,
            z_coordinates_2d=None,
            distance_step_3d=2 * units.m,
            widths_3d=None,
            z_step_3d=2 * units.m,
            passband=None,
            min_antenna_distance=5. * units.m,
            debug_folder='.'
    ):
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
        template: BaseTrace object or object of child class
            An electric field to be used to calculate the template which are correlated
            with the channel traces in order to determine the timing difference between
            channels
        distances_2d: array of float
            A list of horizontal distances from the center of the station at which the first
            rough scan to determine the search volume is done. The minimum and maximum of this
            list is later also used as the minimum and maximum distance for the finer search
        azimuths_2d: array of float
            Array of azimuths to be used in the first scan to determine the search volume
        z_coordinates_2d: array of float
            Array of the z coordinates relative to the surface to be used in the first scan
            to determine the search volume. The maximum depth is also used as the maximum
            depth for the finer search
        distance_step_3d: float
            Step size for the horizontal distances used in the finer scan
        widths_3d: array of float
            List of distances to the left and right of the line determined in the first rough
            scan on which the finer scan should be performed
        z_step_3d: float
            Step size for the depts used in the finer scan
        passband: array of float
            Lower and upper bounds off the bandpass filter that is applied to the channel
            waveform and the template before the correlations are determined. This filter
            does not affect the voltages stored in the channels.
        min_antenna_distance: float
            Minimum distance two antennas need to have to be used as a pair in the reconstruction
        debug_folder: string
            Path to the folder in which debug plots should be saved if the debug=True option
            is picked in the run() method.

        """

        self.__detector = detector
        self.__channel_ids = channel_ids
        self.__station_id = station_id
        self.__debug_folder = debug_folder
        self.__channel_pairs = []
        for i in range(len(channel_ids) - 1):
            for j in range(i + 1, len(channel_ids)):
                relative_positions = detector.get_relative_position(station_id, channel_ids[i]) - detector.get_relative_position(station_id, channel_ids[j])
                if detector.get_antenna_type(station_id, channel_ids[i]) == detector.get_antenna_type(station_id, channel_ids[j]) \
                        and np.sqrt(np.sum(relative_positions**2)) > min_antenna_distance:
                    self.__channel_pairs.append([channel_ids[i], channel_ids[j]])
        self.__lookup_table = {}
        self.__header = {}
        self.__electric_field_template = template
        self.__sampling_rate = template.get_sampling_rate()
        self.__passband = passband
        if distances_2d is None:
            self.__distances_2d = np.arange(100, 3000, 200)
        else:
            self.__distances_2d = distances_2d
        if azimuths_2d is None:
            self.__azimuths_2d = np.arange(0, 360.1, 2.5) * units.deg
        else:
            self.__azimuths_2d = azimuths_2d
        if z_coordinates_2d is None:
            self.__z_coordinates_2d = np.arange(-2700, -100, 25)
        else:
            self.__z_coordinates_2d = z_coordinates_2d
        self.__distance_step_3d = distance_step_3d
        if widths_3d is None:
            self.__widths_3d = np.arange(-50, 50, 2.)
        else:
            self.__widths_3d = widths_3d
        self.__z_step_3d = z_step_3d
        for channel_id in channel_ids:
            channel_z = abs(detector.get_relative_position(station_id, channel_id)[2])
            if channel_z not in self.__lookup_table.keys():
                f = NuRadioReco.utilities.io_utilities.read_pickle('{}/lookup_table_{}.p'.format(self.__lookup_table_location, int(abs(channel_z))))
                self.__header[int(channel_z)] = f['header']
                self.__lookup_table[int(abs(channel_z))] = f['antenna_{}'.format(channel_z)]

    def run(
            self,
            event,
            station,
            det,
            debug=False
    ):
        azimuth_grid_2d, z_grid_2d = np.meshgrid(self.__azimuths_2d, self.__z_coordinates_2d)
        distance_correlations = np.zeros(self.__distances_2d.shape)
        full_correlations = np.zeros((len(self.__distances_2d), len(self.__z_coordinates_2d), len(self.__azimuths_2d)))

        if debug:
            plt.close('all')
            fig1 = plt.figure(figsize=(12, (len(self.__channel_pairs) // 2 + len(self.__channel_pairs) % 2)))
        self.__pair_correlations = np.zeros((len(self.__channel_pairs), station.get_channel(self.__channel_ids[0]).get_number_of_samples() + self.__electric_field_template.get_number_of_samples() - 1))
        for i_pair, channel_pair in enumerate(self.__channel_pairs):
            channel_1 = station.get_channel(channel_pair[0])
            channel_2 = station.get_channel(channel_pair[1])
            antenna_response = trace_utilities.get_efield_antenna_factor(
                station=station,
                frequencies=self.__electric_field_template.get_frequencies(),
                channels=[channel_pair[0]],
                detector=det,
                zenith=90. * units.deg,
                azimuth=0,
                antenna_pattern_provider=self.__antenna_pattern_provider
            )[0]
            voltage_spec = (
                antenna_response[0] * self.__electric_field_template.get_frequency_spectrum() + antenna_response[1] * self.__electric_field_template.get_frequency_spectrum()
            ) * det.get_amplifier_response(station.get_id(), channel_pair[0], self.__electric_field_template.get_frequencies())
            if self.__passband is not None:
                voltage_spec *= bandpass_filter.get_filter_response(self.__electric_field_template.get_frequencies(), self.__passband, 'butterabs', 10)
            voltage_template = fft.freq2time(voltage_spec, self.__sampling_rate)
            voltage_template /= np.max(np.abs(voltage_template))
            if self.__passband is None:
                corr_1 = np.abs(hp.get_normalized_xcorr(channel_1.get_trace(), voltage_template))
                corr_2 = np.abs(hp.get_normalized_xcorr(channel_2.get_trace(), voltage_template))
            else:
                corr_1 = np.abs(hp.get_normalized_xcorr(channel_1.get_filtered_trace(self.__passband, 'butterabs', 10), voltage_template))
                corr_2 = np.abs(hp.get_normalized_xcorr(channel_2.get_filtered_trace(self.__passband, 'butterabs', 10), voltage_template))
            correlation_product = np.zeros_like(corr_1)
            sample_shifts = np.arange(-len(corr_1) // 2, len(corr_1) // 2, dtype=int)
            toffset = sample_shifts / channel_1.get_sampling_rate()
            for i_shift, shift_sample in enumerate(sample_shifts):
                correlation_product[i_shift] = np.max((corr_1 * np.roll(corr_2, shift_sample)))
            self.__pair_correlations[i_pair] = correlation_product
            if debug:
                ax1_1 = fig1.add_subplot(len(self.__channel_pairs) // 2 + len(self.__channel_pairs) % 2, 2,
                                         i_pair + 1)
                ax1_1.grid()
                ax1_1.plot(toffset, correlation_product)
        if debug:
            fig1.tight_layout()
            fig1.savefig('{}/{}_{}_correlation.png'.format(self.__debug_folder, event.get_run_number(), event.get_id()))
        for i_dist, distance in enumerate(self.__distances_2d):
            self.__current_distance = distance
            correlation_sum = np.zeros_like(azimuth_grid_2d)

            for i_pair, channel_pair in enumerate(self.__channel_pairs):
                self.__correlation = self.__pair_correlations[i_pair]
                self.__channel_pair = channel_pair
                self.__channel_positions = [self.__detector.get_relative_position(self.__station_id, channel_pair[0]),
                                            self.__detector.get_relative_position(self.__station_id, channel_pair[1])]
                correlation_map = np.zeros_like(correlation_sum)
                for i_ray in range(len(self.__ray_types)):
                    self.__current_ray_types = self.__ray_types[i_ray]
                    correlation_map = np.maximum(self.get_correlation_array_2d(azimuth_grid_2d, z_grid_2d), correlation_map)
                correlation_sum += correlation_map
            distance_correlations[i_dist] = np.max(correlation_sum)
            full_correlations[i_dist] = correlation_sum

        corr_fit_threshold = .7 * np.max(full_correlations)
        flattened_corr = np.max(full_correlations, axis=2).T
        i_max_d = np.argmax(flattened_corr, axis=0)
        corr_mask_d = np.max(flattened_corr, axis=0) > corr_fit_threshold
        line_fit_d = np.polyfit(
            self.__distances_2d[corr_mask_d],
            self.__z_coordinates_2d[i_max_d][corr_mask_d],
            1
        )
        residuals_d = np.sum((self.__z_coordinates_2d[i_max_d][corr_mask_d] - self.__distances_2d[corr_mask_d] * line_fit_d[0] - line_fit_d[1])**2) / np.sum(corr_mask_d.astype(int))
        i_max_z = np.argmax(flattened_corr, axis=1)
        corr_mask_z = np.max(flattened_corr, axis=1) > corr_fit_threshold
        line_fit_z = np.polyfit(
            self.__distances_2d[i_max_z][corr_mask_z],
            self.__z_coordinates_2d[corr_mask_z],
            1
        )
        residuals_z = np.sum((self.__z_coordinates_2d[corr_mask_z] - self.__distances_2d[i_max_z][corr_mask_z] * line_fit_z[0] - line_fit_z[1])**2) / np.sum(corr_mask_z.astype(int))
        if residuals_d <= residuals_z:
            slope = line_fit_d[0]
            offset = line_fit_d[1]
            max_z_offset = np.max([50, np.min([200, np.max(self.__z_coordinates_2d[i_max_d][corr_mask_d] - self.__distances_2d[corr_mask_d] * slope - offset)])])
            min_z_offset = np.max([50, np.min([200, np.max(-self.__z_coordinates_2d[i_max_d][corr_mask_d] + self.__distances_2d[corr_mask_d] * slope + offset)])])
            flattened_corr_theta = np.max(full_correlations, axis=1)
            theta_corr_mask = np.max(flattened_corr_theta, axis=1) >= corr_fit_threshold
            i_max_theta = np.argmax(flattened_corr_theta, axis=1)
            median_theta = np.median(self.__azimuths_2d[i_max_theta][theta_corr_mask])
            z_fit = False
        else:
            slope = line_fit_z[0]
            offset = line_fit_z[1]
            max_z_offset = np.max([50, np.min([200, np.max(self.__z_coordinates_2d[corr_mask_z] - self.__distances_2d[i_max_z][corr_mask_z] * slope - offset)])])
            min_z_offset = np.max([50, np.min([200, np.max(-self.__z_coordinates_2d[corr_mask_z] + self.__distances_2d[i_max_z][corr_mask_z] * slope + offset)])])
            flattened_corr_theta = np.max(full_correlations, axis=0)
            theta_corr_mask = np.max(flattened_corr_theta, axis=1) >= corr_fit_threshold
            i_max_theta = np.argmax(flattened_corr_theta, axis=1)
            median_theta = np.median(self.__azimuths_2d[i_max_theta][theta_corr_mask])
            z_fit = True
        if debug:
            self.__draw_2d_correlation_map(event, full_correlations)
            self.__draw_search_zones(
                event,
                slope,
                offset,
                line_fit_d,
                line_fit_z,
                min_z_offset,
                max_z_offset,
                i_max_d,
                i_max_z,
                corr_mask_d,
                corr_mask_z,
                z_fit,
                i_max_theta,
                theta_corr_mask,
                median_theta
            )

        # <--- 3D Fit ---> #
        distances_3d = np.arange(self.__distances_2d[0], self.__distances_2d[-1], self.__distance_step_3d)
        z_coords = slope * distances_3d + offset
        distances_3d = distances_3d[(z_coords < 0) & (z_coords > -2700)]
        search_heights = np.arange(-1.1 * min_z_offset, 1.1 * max_z_offset, self.__z_step_3d)
        x_0, y_0, z_0 = np.meshgrid(distances_3d, self.__widths_3d, search_heights)

        z_coords = z_0 + slope * x_0 + offset
        x_coords = np.cos(median_theta) * x_0 - y_0 * np.sin(median_theta)
        y_coords = np.sin(median_theta) * x_0 + y_0 * np.cos(median_theta)

        correlation_sum = np.zeros_like(z_coords)

        for i_pair, channel_pair in enumerate(self.__channel_pairs):
            self.__correlation = self.__pair_correlations[i_pair]
            self.__channel_pair = channel_pair
            self.__channel_positions = [self.__detector.get_relative_position(self.__station_id, channel_pair[0]),
                                        self.__detector.get_relative_position(self.__station_id, channel_pair[1])]
            correlation_map = np.zeros_like(correlation_sum)
            for i_ray in range(len(self.__ray_types)):
                self.__current_ray_types = self.__ray_types[i_ray]
                correlation_map = np.maximum(self.get_correlation_array_3d(x_coords, y_coords, z_coords), correlation_map)
            correlation_sum += correlation_map
        i_max = np.unravel_index(np.argmax(correlation_sum), correlation_sum.shape)
        if debug:
            self.__draw_vertex_reco(
                event,
                correlation_sum / np.max(correlation_sum),
                x_0,
                y_0,
                z_0,
                x_coords,
                y_coords,
                z_coords,
                slope,
                offset,
                median_theta,
                i_max
            )
        # <<--- DnR Reco --->> #
        self.__self_correlations = np.zeros((len(self.__channel_ids), station.get_channel(self.__channel_ids[0]).get_number_of_samples() + self.__electric_field_template.get_number_of_samples() - 1))
        self_correlation_sum = np.zeros_like(z_coords)
        for i_channel, channel_id in enumerate(self.__channel_ids):
            channel = station.get_channel(channel_id)
            antenna_response = trace_utilities.get_efield_antenna_factor(
                station=station,
                frequencies=self.__electric_field_template.get_frequencies(),
                channels=[channel_id],
                detector=det,
                zenith=90. * units.deg,
                azimuth=0,
                antenna_pattern_provider=self.__antenna_pattern_provider
            )[0]
            voltage_spec = (
                antenna_response[0] * self.__electric_field_template.get_frequency_spectrum() + antenna_response[1] * self.__electric_field_template.get_frequency_spectrum()
            ) * det.get_amplifier_response(station.get_id(), channel_id, self.__electric_field_template.get_frequencies())
            if self.__passband is not None:
                voltage_spec *= bandpass_filter.get_filter_response(self.__electric_field_template.get_frequencies(), self.__passband, 'butter', 10)
            voltage_template = fft.freq2time(voltage_spec, self.__sampling_rate)
            voltage_template /= np.max(np.abs(voltage_template))
            if self.__passband is None:
                corr_1 = hp.get_normalized_xcorr(channel.get_trace(), voltage_template)
                corr_2 = hp.get_normalized_xcorr(channel.get_trace(), voltage_template)
            else:
                corr_1 = np.abs(hp.get_normalized_xcorr(channel.get_filtered_trace(self.__passband, 'butter', 10), voltage_template))
                corr_2 = np.abs(hp.get_normalized_xcorr(channel.get_filtered_trace(self.__passband, 'butter', 10), voltage_template))
            correlation_product = np.zeros_like(corr_1)
            sample_shifts = np.arange(-len(corr_1) // 2, len(corr_1) // 2, dtype=int)
            toffset = sample_shifts / channel.get_sampling_rate()
            for i_shift, shift_sample in enumerate(sample_shifts):
                correlation_product[i_shift] = np.max((corr_1 * np.roll(corr_2, shift_sample)))
            correlation_product = np.abs(correlation_product)
            correlation_product[np.abs(toffset) < 20] = 0

            self.__correlation = correlation_product
            self.__channel_pair = [channel_id, channel_id]
            self.__channel_positions = [self.__detector.get_relative_position(self.__station_id, channel_id),
                                        self.__detector.get_relative_position(self.__station_id, channel_id)]
            correlation_map = np.zeros_like(correlation_sum)
            for i_ray in range(len(self.__ray_types)):
                if self.__ray_types[i_ray][0] != self.__ray_types[i_ray][1]:
                    self.__current_ray_types = self.__ray_types[i_ray]
                    correlation_map = np.maximum(self.get_correlation_array_3d(x_coords, y_coords, z_coords), correlation_map)
            self_correlation_sum += correlation_map
        combined_correlations = correlation_sum / len(self.__channel_pairs) + self_correlation_sum / len(self.__channel_ids)
        i_max_dnr = np.unravel_index(np.argmax(combined_correlations), combined_correlations.shape)
        vertex_x = x_coords[i_max_dnr]
        vertex_y = y_coords[i_max_dnr]
        vertex_z = z_coords[i_max_dnr]
        station.set_parameter(stnp.nu_vertex, np.array([vertex_x, vertex_y, vertex_z]))
        if debug:
            self.__draw_dnr_reco(
                event,
                correlation_sum / np.max(correlation_sum),
                self_correlation_sum / np.max(self_correlation_sum),
                combined_correlations / np.max(combined_correlations),
                x_0,
                y_0,
                z_0,
                slope,
                offset,
                median_theta,
                i_max,
                i_max_dnr
            )

    def get_correlation_array_2d(self, phi, z):
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
        x = self.__current_distance * np.cos(phi)
        y = self.__current_distance * np.sin(phi)
        d_hor1 = np.sqrt((x - channel_pos1[0])**2 + (y - channel_pos1[1])**2)
        d_hor2 = np.sqrt((x - channel_pos2[0])**2 + (y - channel_pos2[1])**2)
        res = self.get_correlation_for_pos(np.array([d_hor1, d_hor2]), z)
        return res

    def get_correlation_array_3d(self, x, y, z):
        channel_pos1 = self.__channel_positions[0]
        channel_pos2 = self.__channel_positions[1]
        d_hor1 = np.sqrt((x - channel_pos1[0])**2 + (y - channel_pos1[1])**2)
        d_hor2 = np.sqrt((x - channel_pos2[0])**2 + (y - channel_pos2[1])**2)
        res = self.get_correlation_for_pos(np.array([d_hor1, d_hor2]), z)
        res[np.abs(res) < .8 * np.max(np.abs(res))] = 0
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
        corr_index[np.isnan(delta_t)] = 0
        mask = (corr_index > 0) & (corr_index < self.__correlation.shape[0]) & (~np.isinf(delta_t))
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
        i_z = np.array(np.round((z - self.__header[channel_type]['z_min']) / self.__header[channel_type]['d_z'])).astype(int)
        i_x_1 = np.array(np.floor((d_hor - self.__header[channel_type]['x_min']) / self.__header[channel_type]['d_x'])).astype(int)
        cell_dist_1 = i_x_1 * self.__header[channel_type]['d_x'] + self.__header[channel_type]['x_min']
        mask[i_x_1 > self.__lookup_table[channel_type][ray_type].shape[0] - 1] = False
        mask[i_z > self.__lookup_table[channel_type][ray_type].shape[1] - 1] = False
        i_x_1[~mask] = 0
        i_z[~mask] = 0
        travel_times_1 = self.__lookup_table[channel_type][ray_type][(i_x_1, i_z)]
        i_x_2 = np.array(np.ceil((d_hor - self.__header[channel_type]['x_min']) / self.__header[channel_type]['d_x'])).astype(int)
        cell_dist_2 = i_x_2 * self.__header[channel_type]['d_x'] + self.__header[channel_type]['x_min']
        i_x_2[~mask] = 0
        travel_times_2 = self.__lookup_table[channel_type][ray_type][(i_x_2, i_z)]
        slopes = np.zeros_like(travel_times_1)
        slopes[i_x_2 > i_x_1] = (travel_times_1 - travel_times_2)[i_x_2 > i_x_1] / (cell_dist_1 - cell_dist_2)[i_x_2 > i_x_1]
        travel_times = (d_hor - cell_dist_1) * slopes + travel_times_1
        travel_times[~mask] = np.nan
        return travel_times

    def __draw_2d_correlation_map(self, event, correlation_map):
        fig4 = plt.figure(figsize=(4, 12))
        ax4_1 = fig4.add_subplot(311)
        d_0, z_0 = np.meshgrid(self.__distances_2d, self.__z_coordinates_2d)
        ax4_1.pcolor(
            d_0,
            z_0,
            np.max(correlation_map, axis=2).T
        )
        ax4_1.grid()
        theta_0, d_0 = np.meshgrid(self.__azimuths_2d, self.__distances_2d)
        ax4_2 = fig4.add_subplot(312, projection='polar')
        ax4_2.pcolor(
            theta_0,
            d_0,
            np.max(correlation_map, axis=1)
        )
        ax4_2.grid()
        ax4_3 = fig4.add_subplot(313)
        theta_0, z_0 = np.meshgrid(self.__azimuths_2d, self.__z_coordinates_2d)
        ax4_3.pcolor(
            theta_0 / units.deg,
            z_0,
            np.max(correlation_map, axis=0)
        )
        ax4_3.grid()
        sim_vertex = None
        for sim_shower in event.get_sim_showers():
            sim_vertex = sim_shower.get_parameter(shp.vertex)
            break
        if sim_vertex is not None:
            ax4_1.scatter(
                [np.sqrt(sim_vertex[0] ** 2 + sim_vertex[1] ** 2)],
                [sim_vertex[2]],
                c='r',
                alpha=.5,
                marker='+'
            )
            ax4_2.scatter(
                [hp.cartesian_to_spherical(sim_vertex[0], sim_vertex[1], sim_vertex[2])[1]],
                [np.sqrt(sim_vertex[0] ** 2 + sim_vertex[1] ** 2)],
                c='r',
                alpha=.5,
                marker='+'
            )
            ax4_3.scatter(
                [hp.get_normalized_angle(
                    hp.cartesian_to_spherical(sim_vertex[0], sim_vertex[1], sim_vertex[2])[1]) / units.deg],
                [(sim_vertex[2])],
                c='r',
                alpha=.5,
                marker='+'
            )
        ax4_1.set_xlabel('d [m]')
        ax4_1.set_ylabel('z [m]')
        ax4_3.set_xlabel(r'$\phi [^\circ]$')
        ax4_3.set_ylabel('z [m]')
        fig4.tight_layout()
        fig4.savefig('{}/{}_{}_2D_correlation_maps.png'.format(self.__debug_folder, event.get_run_number(), event.get_id()))

    def __draw_search_zones(
            self,
            event,
            slope,
            offset,
            line_fit_d,
            line_fit_z,
            min_z_offset,
            max_z_offset,
            i_max_d,
            i_max_z,
            corr_mask_d,
            corr_mask_z,
            z_fit,
            i_max_theta,
            theta_corr_mask,
            median_theta
    ):
        fig5 = plt.figure(figsize=(8, 8))
        ax5_1_1 = fig5.add_subplot(221)
        ax5_1_1.grid()
        ax5_1_2 = fig5.add_subplot(222)
        ax5_1_2.grid()
        ax5_1_1.fill_between(
            self.__distances_2d,
            self.__distances_2d * slope + offset + 1.1 * max_z_offset,
            self.__distances_2d * slope + offset - 1.1 * min_z_offset,
            color='k',
            alpha=.2
        )
        ax5_1_1.scatter(
            self.__distances_2d[corr_mask_d],
            self.__z_coordinates_2d[i_max_d][corr_mask_d]
        )
        ax5_1_1.scatter(
            self.__distances_2d[~corr_mask_d],
            self.__z_coordinates_2d[i_max_d][~corr_mask_d],
            c='k',
            alpha=.5
        )
        ax5_1_1.plot(
            self.__distances_2d,
            self.__distances_2d * slope + offset,
            color='k',
            linestyle=':'
        )
        ax5_1_1.fill_between(
            self.__distances_2d,
            self.__distances_2d * slope + offset + 1.1 * max_z_offset,
            self.__distances_2d * slope + offset - 1.1 * min_z_offset,
            color='k',
            alpha=.2
        )
        ax5_1_1.plot(
            self.__distances_2d,
            self.__distances_2d * line_fit_d[0] + line_fit_d[1],
            color='r',
            linestyle=':'
        )
        ax5_1_1.fill_between(
            self.__distances_2d,
            self.__distances_2d * line_fit_d[0] + line_fit_d[1] + 1.1 * max_z_offset,
            self.__distances_2d * line_fit_d[0] + line_fit_d[1] - 1.1 * min_z_offset,
            color='r',
            alpha=.2
        )
        ax5_1_2.scatter(
            self.__distances_2d[i_max_z][corr_mask_z],
            self.__z_coordinates_2d[corr_mask_z]
        )
        ax5_1_2.scatter(
            self.__distances_2d[i_max_z][~corr_mask_z],
            self.__z_coordinates_2d[~corr_mask_z],
            c='k',
            alpha=.5
        )
        ax5_1_2.plot(
            self.__distances_2d,
            self.__distances_2d * slope + offset,
            color='k',
            linestyle=':'
        )
        ax5_1_2.plot(
            self.__distances_2d,
            self.__distances_2d * line_fit_z[0] + line_fit_z[1],
            color='r',
            linestyle=':'
        )
        ax5_1_2.fill_between(
            self.__distances_2d,
            self.__distances_2d * line_fit_z[0] + line_fit_z[1] + 1.1 * max_z_offset,
            self.__distances_2d * line_fit_z[0] + line_fit_z[1] - 1.1 * min_z_offset,
            color='r',
            alpha=.2
        )
        ax5_1_2.plot(
            self.__distances_2d,
            self.__distances_2d * slope + offset,
            color='k',
            linestyle=':'
        )
        ax5_1_2.fill_between(
            self.__distances_2d,
            self.__distances_2d * slope + offset + 1.1 * max_z_offset,
            self.__distances_2d * slope + offset - 1.1 * min_z_offset,
            color='k',
            alpha=.2
        )

        if z_fit:
            ax5_2 = fig5.add_subplot(2, 2, (3, 4))
            ax5_2.grid()
        else:
            ax5_2 = fig5.add_subplot(2, 2, (3, 4), projection='polar')
        if z_fit:
            ax5_2.scatter(
                self.__azimuths_2d[i_max_theta][theta_corr_mask],
                np.abs(self.__z_coordinates_2d)[theta_corr_mask]
            )
            ax5_2.scatter(
                self.__azimuths_2d[i_max_theta][~theta_corr_mask],
                np.abs(self.__z_coordinates_2d)[~theta_corr_mask],
                c='k',
                alpha=.5
            )
            ax5_2.plot(
                [median_theta, median_theta],
                [np.abs(self.__z_coordinates_2d[0]), np.abs(self.__z_coordinates_2d[-1])],
                color='k',
                linestyle=':'
            )
        else:
            ax5_2.scatter(
                self.__azimuths_2d[i_max_theta][theta_corr_mask],
                self.__distances_2d[theta_corr_mask]
            )
            ax5_2.scatter(
                self.__azimuths_2d[i_max_theta][~theta_corr_mask],
                self.__distances_2d[~theta_corr_mask],
                c='k',
                alpha=.5
            )
            ax5_2.plot(
                [median_theta, median_theta],
                [self.__distances_2d[0], self.__distances_2d[-1]],
                color='k',
                linestyle=':'
            )
        sim_vertex = None
        for sim_shower in event.get_sim_showers():
            sim_vertex = sim_shower.get_parameter(shp.vertex)
            break
        if sim_vertex is not None:
            ax5_1_1.axhline(sim_vertex[2], color='r', linestyle='--', alpha=.5)
            ax5_1_1.axvline(np.sqrt(sim_vertex[0] ** 2 + sim_vertex[1] ** 2), color='r', linestyle='--', alpha=.5)
            ax5_1_2.axhline(sim_vertex[2], color='r', linestyle='--', alpha=.5)
            ax5_1_2.axvline(np.sqrt(sim_vertex[0] ** 2 + sim_vertex[1] ** 2), color='r', linestyle='--', alpha=.5)
            ax5_2.scatter(
                [hp.cartesian_to_spherical(sim_vertex[0], sim_vertex[1], sim_vertex[2])[1]],
                [np.sqrt(sim_vertex[0] ** 2 + sim_vertex[1] ** 2)],
                c='r',
                alpha=.5,
                marker='+'
            )
        ax5_1_1.set_xlabel('d [m]')
        ax5_1_2.set_xlabel('d [m]')
        ax5_1_1.set_ylabel('z [m]')
        ax5_1_2.set_ylabel('z [m]')
        if z_fit:
            ax5_2.set_xlabel(r'$\phi [^\circ]$')
            ax5_2.set_ylabel('|z| [m]')
        fig5.tight_layout()
        fig5.savefig('{}/{}_{}_search_zones.png'.format(self.__debug_folder, event.get_run_number(), event.get_id()))

    def __draw_vertex_reco(
            self,
            event,
            correlation_sum,
            x_0,
            y_0,
            z_0,
            x_coords,
            y_coords,
            z_coords,
            slope,
            offset,
            median_theta,
            i_max
    ):
        fig6 = plt.figure(figsize=(8, 16))
        ax6_1 = fig6.add_subplot(411)
        vmin = .6
        vmax = 1.
        colormap = cm.get_cmap('viridis', 16)
        sim_vertex = None
        for sim_shower in event.get_sim_showers():
            sim_vertex = sim_shower.get_parameter(shp.vertex)
            break
        cplot1 = ax6_1.pcolor(
            x_0[0],
            z_0[0],
            np.max(correlation_sum, axis=0),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(cplot1, ax=ax6_1)
        ax6_2 = fig6.add_subplot(412)
        cplot2 = ax6_2.pcolor(
            x_0[:, :, 0],
            y_0[:, :, 0],
            np.max(correlation_sum, axis=2),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(cplot2, ax=ax6_2)

        ax6_3 = fig6.add_subplot(413)
        cplot3 = ax6_3.pcolor(
            x_0[0],
            z_coords[0],
            np.max(correlation_sum, axis=0),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(cplot3, ax=ax6_3)
        ax6_3.set_aspect('equal')
        ax6_4 = fig6.add_subplot(414)
        cplot4 = ax6_4.pcolor(
            x_coords[:, :, 0],
            y_coords[:, :, 0],
            np.max(correlation_sum, axis=2),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(cplot4, ax=ax6_4)
        ax6_4.set_aspect('equal')
        if sim_vertex is not None:
            sim_vertex_dhor = np.sqrt(sim_vertex[0] ** 2 + sim_vertex[1] ** 2)
            ax6_1.scatter(
                [sim_vertex_dhor],
                [sim_vertex[2] - sim_vertex_dhor * slope - offset],
                c='r',
                marker='+'
            )
            ax6_2.scatter(
                [np.cos(median_theta) * sim_vertex[0] + np.sin(median_theta) * sim_vertex[1]],
                [-np.sin(median_theta) * sim_vertex[0] + np.cos(median_theta) * sim_vertex[1]],
                c='r',
                marker='+'
            )
            ax6_3.scatter(
                [sim_vertex_dhor],
                [sim_vertex[2]],
                c='r',
                marker='+'
            )
            ax6_4.scatter(
                [sim_vertex[0]],
                [sim_vertex[1]],
                c='r',
                marker='+'
            )
        ax6_1.scatter(
            [x_0[i_max]],
            [z_0[i_max]],
            c='k',
            marker='+'
        )
        ax6_2.scatter(
            [x_0[i_max]],
            [y_0[i_max]],
            c='k',
            marker='+'
        )
        ax6_3.scatter(
            [x_0[i_max]],
            [z_coords[i_max]],
            c='k',
            marker='+'
        )
        ax6_4.scatter(
            [x_coords[i_max]],
            [y_coords[i_max]],
            c='k',
            marker='+'
        )
        ax6_1.set_xlabel('r [m]')
        ax6_1.set_ylabel(r'$\Delta z$ [m]')
        ax6_2.set_xlabel(r'$\Delta x$ [m]')
        ax6_2.set_ylabel(r'$\Delta y$ [m]')
        ax6_3.set_xlabel('r [m]')
        ax6_3.set_ylabel('z [m]')
        ax6_4.set_xlabel('x [m]')
        ax6_4.set_ylabel('y [m]')
        fig6.tight_layout()
        fig6.savefig('{}/{}_{}_slices.png'.format(self.__debug_folder, event.get_run_number(), event.get_id()))

    def __draw_dnr_reco(
            self,
            event,
            correlation_sum,
            self_correlation_sum,
            combined_correlations,
            x_0,
            y_0,
            z_0,
            slope,
            offset,
            median_theta,
            i_max,
            i_max_dnr
    ):
        fig8 = plt.figure(figsize=(12, 8))
        ax8_1 = fig8.add_subplot(232)
        ax8_2 = fig8.add_subplot(235)
        ax8_3 = fig8.add_subplot(233)
        ax8_4 = fig8.add_subplot(236)
        ax8_5 = fig8.add_subplot(231)
        ax8_6 = fig8.add_subplot(234)
        colormap = cm.get_cmap('viridis', 16)
        vmin = .6
        vmax = 1.
        cplot1 = ax8_1.pcolor(
            x_0[0],
            z_0[0],
            np.max(self_correlation_sum, axis=0),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(cplot1, ax=ax8_1)
        cplot2 = ax8_2.pcolor(
            x_0[:, :, 0],
            y_0[:, :, 0],
            np.max(self_correlation_sum, axis=2),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(cplot2, ax=ax8_2)

        cplot3 = ax8_3.pcolor(
            x_0[0],
            z_0[0],
            np.max(combined_correlations, axis=0),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(cplot3, ax=ax8_3)
        cplot4 = ax8_4.pcolor(
            x_0[:, :, 0],
            y_0[:, :, 0],
            np.max(combined_correlations, axis=2),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(cplot4, ax=ax8_4)
        cplot5 = ax8_5.pcolor(
            x_0[0],
            z_0[0],
            np.max(correlation_sum, axis=0),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(cplot5, ax=ax8_5)
        cplot6 = ax8_6.pcolor(
            x_0[:, :, 0],
            y_0[:, :, 0],
            np.max(correlation_sum, axis=2),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        ax8_5.scatter(
            [x_0[i_max]],
            [z_0[i_max]],
            c='k',
            marker='+'
        )
        ax8_3.scatter(
            [x_0[i_max_dnr]],
            [z_0[i_max_dnr]],
            c='k',
            marker='+'
        )
        plt.colorbar(cplot6, ax=ax8_6)
        sim_vertex = None
        for sim_shower in event.get_sim_showers():
            sim_vertex = sim_shower.get_parameter(shp.vertex)
            break
        if sim_vertex is not None:
            sim_vertex_dhor = np.sqrt(sim_vertex[0] ** 2 + sim_vertex[1] ** 2)
            ax8_1.scatter(
                [sim_vertex_dhor],
                [sim_vertex[2] - sim_vertex_dhor * slope - offset],
                c='r',
                marker='+'
            )
            ax8_2.scatter(
                [np.cos(median_theta) * sim_vertex[0] + np.sin(median_theta) * sim_vertex[1]],
                [-np.sin(median_theta) * sim_vertex[0] + np.cos(median_theta) * sim_vertex[1]],
                c='r',
                marker='+'
            )
            ax8_3.scatter(
                [sim_vertex_dhor],
                [sim_vertex[2] - sim_vertex_dhor * slope - offset],
                c='r',
                marker='+'
            )
            ax8_4.scatter(
                [np.cos(median_theta) * sim_vertex[0] + np.sin(median_theta) * sim_vertex[1]],
                [-np.sin(median_theta) * sim_vertex[0] + np.cos(median_theta) * sim_vertex[1]],
                c='r',
                marker='+'
            )
            ax8_5.scatter(
                [sim_vertex_dhor],
                [sim_vertex[2] - sim_vertex_dhor * slope - offset],
                c='r',
                marker='+'
            )
            ax8_6.scatter(
                [np.cos(median_theta) * sim_vertex[0] + np.sin(median_theta) * sim_vertex[1]],
                [-np.sin(median_theta) * sim_vertex[0] + np.cos(median_theta) * sim_vertex[1]],
                c='r',
                marker='+'
            )
        ax8_1.grid()
        ax8_2.grid()
        fig8.tight_layout()
        fig8.savefig('{}/{}_{}_dnr_reco.png'.format(self.__debug_folder, event.get_run_number(), event.get_id()))
