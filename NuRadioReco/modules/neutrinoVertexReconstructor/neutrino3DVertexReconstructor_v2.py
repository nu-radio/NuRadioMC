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
import scipy.optimize
import scipy.ndimage
# import scipy.interpolate
import logging
import pickle

logging.basicConfig()
logger = logging.getLogger('vertexReco')
logger.setLevel(logging.DEBUG)

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
        # self.__get_time_scipy = {}
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
            ['D', 'D'],
            ['D', 'R'],
            ['R', 'D'],
            ['R', 'R']
        ]
        # self.__ray_types = [
        #     ['direct', 'direct'],
        #     ['reflected', 'reflected'],
        #     ['refracted', 'refracted'],
        #     ['direct', 'reflected'],
        #     ['reflected', 'direct'],
        #     ['direct', 'refracted'],
        #     ['refracted', 'direct'],
        #     ['reflected', 'refracted'],
        #     ['refracted', 'reflected']
        # ]

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
            use_maximum_filter=True,
            resolution_target=0.1  *units.deg,
            debug_folder='.',
            sampling_rate=None,
            debug_formats='.png'
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
        debug_formats: string | list (default: '.png')
            The format(s) to use to save the debug plots. Include '.pickle' to save
            plots in pickle (reopenable / editable in future Python sessions) format

        """

        self.__detector = detector
        self.__channel_ids = channel_ids
        self.__station_id = station_id
        self.__debug_folder = debug_folder
        self._dTheta_goal = resolution_target
        self.__channel_pairs = []
        self._use_maximum_filter = use_maximum_filter
        for i in range(len(channel_ids) - 1):
            for j in range(i + 1, len(channel_ids)):
                relative_positions = detector.get_relative_position(station_id, channel_ids[i]) - detector.get_relative_position(station_id, channel_ids[j])
                if detector.get_antenna_type(station_id, channel_ids[i]) == detector.get_antenna_type(station_id, channel_ids[j]) \
                        and np.sqrt(np.sum(relative_positions**2)) > min_antenna_distance:
                    self.__channel_pairs.append([channel_ids[i], channel_ids[j]])
        self.__lookup_table = {}
        self.__header = {}
        self._dTheta0 = 2 * units.deg
        if hasattr(template, '__len__'): # template is an array-like or a dict
            self.__voltage_trace_template = template
            self.__sampling_rate = sampling_rate # this needs to be defined in this case!
            if sampling_rate is None:
                raise TypeError("``sampling_rate`` needs to be defined if not using an ElectricField class object as the template!")
        else: # default case - template should be of base_trace class
            self.__electric_field_template = template
            self.__sampling_rate = template.get_sampling_rate()
            self.__voltage_trace_template = None
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
            channel_type = int(channel_z)
            if channel_type not in self.__lookup_table.keys():
                f = NuRadioReco.utilities.io_utilities.read_pickle('{}/lookup_table_{}.p'.format(self.__lookup_table_location, int(abs(channel_z))))
                self.__header[channel_type] = f['header']
                self.__lookup_table[channel_type] = f['antenna_{}'.format(channel_z)]
        self.__start_times = dict()
        self.__channel_correlations = dict()
        self.__debug_fmts = debug_formats
        logger.debug("Using {} channels / {} channel pairs".format(len(self.__channel_ids), len(self.__channel_pairs)))

    def run(
            self,
            event,
            station,
            det,
            debug=False
        ):
        """
        Run the vertex reconstruction.

        Parameters
        ----------
        event: Event object
            the event to reconstruct
        station: Station object
            the Station to use for the reconstruction
        det: Detector object
            the `Detector` description for the detector used/simulated
        debug: bool (default: False)
            if True, save debug plots at various stages of the reconstruction
        """

        if debug:
            plt.close('all')
            fig1 = plt.figure(figsize=(12, (len(self.__channel_pairs) + len(self.__channel_pairs) % 2)))
        self.__pair_correlations = np.zeros((len(self.__channel_pairs), station.get_channel(self.__channel_ids[0]).get_number_of_samples() + self.__electric_field_template.get_number_of_samples() - 1))
        # compute pair correlations
        for i_pair, channel_pair in enumerate(self.__channel_pairs):
            channel_1 = station.get_channel(channel_pair[0])
            channel_2 = station.get_channel(channel_pair[1])
            self.__start_times[channel_pair[0]] = channel_1.get_trace_start_time()
            self.__start_times[channel_pair[1]] = channel_2.get_trace_start_time()

            if self.__voltage_trace_template is None:
                antenna_response = trace_utilities.get_efield_antenna_factor(
                    station=station,
                    frequencies=self.__electric_field_template.get_frequencies(),
                    channels=[channel_pair[0]],
                    detector=det,
                    zenith=70. * units.deg,
                    azimuth=0,
                    antenna_pattern_provider=self.__antenna_pattern_provider
                )[0]
                amp_response = det.get_amplifier_response(station.get_id(), channel_pair[0], self.__electric_field_template.get_frequencies())
                voltage_spec = (
                    antenna_response[0] * self.__electric_field_template.get_frequency_spectrum() + antenna_response[1] * self.__electric_field_template.get_frequency_spectrum()
                ) * amp_response
                if self.__passband is not None:
                    voltage_spec *= bandpass_filter.get_filter_response(self.__electric_field_template.get_frequencies(), self.__passband, 'butterabs', 10)
                voltage_template = fft.freq2time(voltage_spec, self.__sampling_rate)
            elif isinstance(self.__voltage_trace_template, dict):
                voltage_template = self.__voltage_trace_template[channel_pair[0]]
            else:
                voltage_template = self.__voltage_trace_template
            voltage_template /= np.max(np.abs(voltage_template))
            if self.__passband is None:
                corr_1 = np.abs(hp.get_normalized_xcorr(channel_1.get_trace(), voltage_template))
                corr_2 = np.abs(hp.get_normalized_xcorr(channel_2.get_trace(), voltage_template))
            else:
                corr_1 = np.abs(hp.get_normalized_xcorr(channel_1.get_filtered_trace(self.__passband, 'butterabs', 10), voltage_template))
                corr_2 = np.abs(hp.get_normalized_xcorr(channel_2.get_filtered_trace(self.__passband, 'butterabs', 10), voltage_template))
            correlation_product = np.zeros_like(corr_1)
            sample_shifts = np.arange(-len(corr_1) // 2, len(corr_1) // 2, dtype=int)
            self.__channel_correlations[channel_pair[0]] = corr_1
            self.__channel_correlations[channel_pair[1]] = corr_2
            toffset = sample_shifts / channel_1.get_sampling_rate()
            for i_shift, shift_sample in enumerate(sample_shifts):
                correlation_product[i_shift] = np.max((corr_1 * np.roll(corr_2, shift_sample)))
            corr_median = np.median(correlation_product)
            corr_diff = correlation_product - corr_median
            correlation_product = np.fmax(corr_diff, 0)
            # correlation_product = np.sign(corr_diff) * corr_diff**2 / corr_median**2
            # correlation_product -= np.min(correlation_product)
            self.__pair_correlations[i_pair] = correlation_product
            if debug:
                ax1_1 = fig1.add_subplot(len(self.__channel_pairs) // 2 + len(self.__channel_pairs) % 2, 2,
                                         i_pair + 1)
                ax1_1.grid()
                ax1_1.plot(toffset, correlation_product)
                ax1_1.set_title('Ch.{} & Ch.{}'.format(channel_pair[0], channel_pair[1]))
                ax1_1.set_xlabel(r'$\Delta t$ [ns]')
                ax1_1.set_ylabel('correlation')

        # compute DnR correlations
        self.__self_correlations = np.zeros((len(self.__channel_ids), station.get_channel(self.__channel_ids[0]).get_number_of_samples() + self.__electric_field_template.get_number_of_samples() - 1))
        for i_channel, channel_id in enumerate(self.__channel_ids):
            channel = station.get_channel(channel_id)
            if self.__voltage_trace_template is None:
                antenna_response = trace_utilities.get_efield_antenna_factor(
                    station=station,
                    frequencies=self.__electric_field_template.get_frequencies(),
                    channels=[channel_id],
                    detector=det,
                    zenith=70. * units.deg,
                    azimuth=0,
                    antenna_pattern_provider=self.__antenna_pattern_provider
                )[0]
                amp_response = det.get_amplifier_response(station.get_id(), channel_id, self.__electric_field_template.get_frequencies())
                voltage_spec = (
                    antenna_response[0] * self.__electric_field_template.get_frequency_spectrum() + antenna_response[1] * self.__electric_field_template.get_frequency_spectrum()
                ) * amp_response
                if self.__passband is not None:
                    voltage_spec *= bandpass_filter.get_filter_response(self.__electric_field_template.get_frequencies(), self.__passband, 'butter', 10)
                voltage_template = fft.freq2time(voltage_spec, self.__sampling_rate)
            elif isinstance(self.__voltage_trace_template, dict):
                voltage_template = self.__voltage_trace_template[channel_id]
            else:
                voltage_template = self.__voltage_trace_template
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
            corr_median = np.median(correlation_product)
            corr_diff = correlation_product - corr_median
            correlation_product = np.fmax(corr_diff, 0)
            # correlation_product = np.sign(corr_diff) * corr_diff**2 / corr_median**2
            # correlation_product -= np.min(correlation_product)
            self.__self_correlations[i_channel] = correlation_product

        if debug:
            fig1.tight_layout()
            fname = '{}/{}_{}_correlation'.format(self.__debug_folder, event.get_run_number(), event.get_id())
            save_fig(fig1, fname, self.__debug_fmts)
            plt.close()
            self.__draw_pair_correlations(event)

            # plot channel-template correlations
            fig2 = plt.figure(figsize=(12, len(self.__channel_correlations) // 2 * 3))
            for i_channel, key in enumerate(self.__channel_correlations):
                ax = fig2.add_subplot(
                    len(self.__channel_correlations) // 2 + len(self.__channel_correlations) % 2,
                    2, i_channel+1
                )
                corr = self.__channel_correlations[key]
                t0 = self.__start_times[key] - len(voltage_template) / (2 * self.__sampling_rate)
                times = np.linspace(t0, t0 + len(corr) / self.__sampling_rate, len(corr), endpoint=False)
                ax.plot(times, corr)
                ax.set_title(f"Ch. {key}")
                ax.set_xlabel('t [ns]')
                ax.set_ylabel('correlation')
            fig2.tight_layout()
            fname = '{}/{}_{}_template_correlation'.format(self.__debug_folder, event.get_run_number(), event.get_id())
            save_fig(fig2, fname, self.__debug_fmts)
            plt.close()

            # plot traces, sim traces and templates
            fig3 = plt.figure(figsize=(12, len(self.__channel_correlations) // 2 * 3))
            for i_channel, key in enumerate(self.__channel_correlations):
                ax = fig3.add_subplot(
                    len(self.__channel_correlations) // 2 + len(self.__channel_correlations) % 2,
                    2, i_channel+1
                )
                corr = self.__channel_correlations[key]
                channel = station.get_channel(key)
                if self.__passband is None:
                    trace = channel.get_trace()
                else:
                    trace = channel.get_filtered_trace(self.__passband, 'butterabs', 10)
                ax.plot(channel.get_times(), trace, label='data trace', lw=1)
                if station.has_sim_station():
                    sim_label = 'simulation'
                    for sim_channel in station.get_sim_station().get_channels_by_channel_id(key):
                        if self.__passband is None:
                            sim_trace = sim_channel.get_trace()
                        else:
                            sim_trace = sim_channel.get_filtered_trace(self.__passband, 'butterabs', 10)
                        ax.plot(
                            sim_channel.get_times(),
                            sim_trace,
                            color='red', label=sim_label, lw=1)
                        sim_label = None
                template_t0 = channel.get_trace_start_time() + (
                    np.argmax(corr) - len(voltage_template) + 1) / self.__sampling_rate
                template_times = np.linspace(
                    template_t0, template_t0 + len(voltage_template) / self.__sampling_rate,
                    len(voltage_template), endpoint=False
                )
                ax.plot(template_times, voltage_template * np.max(abs(trace)), label='template',lw=1)
                ax.legend()
                ax.set_title(f"Ch. {key}")
                ax.set_xlabel("t [ns]")
            fig3.tight_layout()
            fname = '{}/{}_{}_traces'.format(self.__debug_folder, event.get_run_number(), event.get_id())
            save_fig(fig3, fname, self.__debug_fmts)
            plt.close()

        # <--- 3D Fit ---> #
        logger.debug("Starting 3D correlation...")
        self._dTheta0 = 2 * units.deg
        # self._dTheta_goal = .1 * units.deg
        self._dTheta_current = self._dTheta0
        dTheta = self._dTheta0
        thetaphi_mask = None
        n_distance_steps = 100
        i_iteration = 1
        while self._dTheta_current > self._dTheta_goal:
            self._dTheta_current = dTheta
            distances_3d = np.logspace(np.log10(self.__distances_2d[0]), np.log10(self.__distances_2d[-1]), n_distance_steps)
            self._d_distance_current = np.exp(1/(n_distance_steps - 1) * np.log(distances_3d[-1] / distances_3d[0])) - 1
            theta_grid = np.arange(np.pi/2, np.pi, dTheta)
            n_phi_per_theta = np.array([int(np.ceil(2*np.pi/dTheta*np.max([0.01, np.abs(np.sin(theta))]))) for theta in theta_grid])
            cum_phi_per_theta = np.cumsum(n_phi_per_theta)
            n_thetaphi_total = cum_phi_per_theta[-1]

            if thetaphi_mask is None: # for the first iteration, we use the full (unmasked) grid in (theta, phi)
                thetaphi = np.array([
                    (theta, phi)
                    for theta in theta_grid
                    for phi in np.linspace(0, 2*np.pi, int(np.ceil(2*np.pi/dTheta*np.max([0.01, np.abs(np.sin(theta))]))), endpoint=False)
                    ]
                ).T

                # To determine our time resolution, we use the dt to the 'nearest neighbours'
                # Here, we find the corresponding indices

                nn_phi = np.arange(len(thetaphi[0])) + 1
                wrap_indices = cum_phi_per_theta - 1
                nn_phi[wrap_indices] = np.concatenate([[0], cum_phi_per_theta])[:-1]

                nn_theta = np.concatenate([
                    (np.around(np.linspace(
                        cum_phi_per_theta[i],
                        cum_phi_per_theta[i] + n_phi_per_theta[i+1],
                        n_phi_per_theta[i], endpoint=False)))
                    for i in range(len(theta_grid)-1)
                ])
                nn_theta = np.array(nn_theta, dtype=int)
                nn_theta = np.concatenate([nn_theta, np.arange(len(nn_theta), len(thetaphi[0]))])

            else: # for iterations >=2, we search only around the previously found maxima
                cum_phi_per_theta0 = np.concatenate([[0], cum_phi_per_theta])
                i_theta = np.ceil((thetaphi_mask[0]-np.pi/2-1e-8) / dTheta).astype(int)
                mask_ind_0 = np.around(
                    cum_phi_per_theta0[i_theta] + n_phi_per_theta[i_theta] * thetaphi_mask[1] / (2*np.pi)
                ).astype(int)
                mask_mask = i_theta > 0
                mask_ind_up = np.around(
                    cum_phi_per_theta0[i_theta[mask_mask]-1] + n_phi_per_theta[i_theta[mask_mask]-1] * thetaphi_mask[1, mask_mask] / (2*np.pi)
                ).astype(int)
                mask_mask = i_theta < len(theta_grid) - 1
                mask_ind_down = np.around(
                    cum_phi_per_theta0[i_theta[mask_mask]+1] + n_phi_per_theta[i_theta[mask_mask]+1] * thetaphi_mask[1, mask_mask] / (2*np.pi)
                ).astype(int)
                mask = np.concatenate([mask_ind_0, mask_ind_up, mask_ind_down])
                mask = np.unique(np.concatenate([mask-1, mask, mask+1]))
                mask = mask[(mask>=0) & (mask < n_thetaphi_total)]

                # if we don't want to run out of memory for smaller angular resolution,
                # we have to build up our search space per theta. We iterate over all theta in the mask +/- 1
                i_theta_unique = np.unique(i_theta)
                i_theta_unique = np.unique(np.concatenate([i_theta_unique-1, i_theta_unique, i_theta_unique+1]))
                i_theta_unique = i_theta_unique[(i_theta_unique >= 0) & (i_theta_unique < len(theta_grid))]

                thetaphi = np.zeros((2,len(mask))) # the (theta, phi) coordinates to include
                nn_phi = np.zeros(len(mask), dtype=int) # the nearest neighbours in phi
                nn_theta = np.zeros(len(mask), dtype=int) # the nearest neighbours in theta
                current_ind = 0
                for i_theta in i_theta_unique: # now i_theta is a single index
                    theta = theta_grid[i_theta]
                    i_min, i_max = cum_phi_per_theta0[i_theta], cum_phi_per_theta[i_theta]
                    submask = mask[(mask>=i_min) & (mask<i_max)] - i_min
                    n_entries = len(submask)
                    if not n_entries: # this should probably never happen?
                        continue

                    # the phi grid for the given value of theta
                    phi = np.linspace(0, 2*np.pi, int(np.ceil(2*np.pi/dTheta*np.max([0.01, np.abs(np.sin(theta))]))), endpoint=False)

                    # now we find the indices that would correspond to the nearest neighbours in theta, phi
                    # if we looked at the full (non-masked) grid. We later convert those to 'in-mask' indices
                    if i_theta < len(theta_grid) - 1:
                        nn_theta_i = np.around(np.linspace(
                            cum_phi_per_theta[i_theta],
                            cum_phi_per_theta[i_theta] + n_phi_per_theta[i_theta+1],
                            n_phi_per_theta[i_theta], endpoint=False)).astype(int)
                    else:
                        nn_theta_i = np.arange(i_min, i_max)
                    nn_phi_i = np.arange(i_min, i_max) + 1
                    nn_phi_i[-1] = i_min

                    thetaphi[0,current_ind:current_ind+n_entries] = theta
                    thetaphi[1,current_ind:current_ind+n_entries] = phi[submask]

                    nn_theta[current_ind:current_ind+n_entries] = nn_theta_i[submask]
                    nn_phi[current_ind:current_ind+n_entries] = nn_phi_i[submask]

                    current_ind += n_entries

                # Now we check if the nearest-neighbour indices are contained in the masked subset
                # and convert them to in-mask indices. For indices that are not in the masked subset,
                # we just set the nearest neighbour of i to i.
                idx = np.arange(len(thetaphi[0]))
                nn_mask = np.in1d(nn_theta, mask)
                mask_idx_sorted = np.argsort(mask)
                nn_theta[nn_mask] = np.searchsorted(mask, nn_theta[nn_mask], sorter=mask_idx_sorted)
                nn_theta[~nn_mask] = idx[~nn_mask]

                nn_mask = np.in1d(nn_phi, mask)
                mask_idx_sorted = np.argsort(mask)
                nn_phi[nn_mask] = np.searchsorted(mask, nn_phi[nn_mask], sorter=mask_idx_sorted)
                nn_phi[~nn_mask] = idx[~nn_mask]

            self.__nn_phi = nn_phi
            self.__nn_theta = nn_theta
            self.__thetaphi = thetaphi
            self.__event = event

            xyz = np.array([
                r * np.array([
                    np.sin(thetaphi[0]) * np.cos(thetaphi[1]),
                    np.sin(thetaphi[0]) * np.sin(thetaphi[1]),
                    np.cos(thetaphi[0])
                ]) for r in distances_3d
            ])
            xyz = xyz.transpose(1,0,2)

            xyz[2] -= 5 * units.m # we start slightly below the surface

            x_coords, y_coords, z_coords = xyz

            logger.debug(f'Fit iteration {i_iteration} - {z_coords.shape[0]} x {z_coords.shape[1]} points')
            logger.debug(f'Current angular resolution {self._dTheta_current / units.deg:.2f} deg (goal: {self._dTheta_goal / units.deg:.2f})')
            logger.debug(f'Current distance resolution {self._d_distance_current*100:.2f}% (goal: None)')
            correlation_sum = np.zeros_like(z_coords)
            number_of_channel_pairs = np.zeros_like(z_coords, dtype=int)

            for i_pair, channel_pair in enumerate(self.__channel_pairs):
                self.__correlation = self.__pair_correlations[i_pair]
                self.__channel_pair = channel_pair
                self.__channel_positions = [self.__detector.get_relative_position(self.__station_id, channel_pair[0]),
                                            self.__detector.get_relative_position(self.__station_id, channel_pair[1])]
                correlation_map = np.zeros_like(correlation_sum)
                for i_ray in range(len(self.__ray_types)):
                    self.__current_ray_types = self.__ray_types[i_ray]
                    correlation_map = np.maximum(self.get_correlation_array_3d(x_coords, y_coords, z_coords), correlation_map)
                    if np.any(np.isnan(correlation_map)):
                        print(i_pair, channel_pair)
                        print(correlation_map)
                        raise ValueError
                number_of_channel_pairs += correlation_map.astype(bool)
                correlation_sum += correlation_map
            # <<--- DnR Reco --->> #
            logger.debug("Starting DnR correlation...")
            self_correlation_sum = np.zeros_like(z_coords)
            number_of_dnr_channels = np.zeros_like(z_coords, dtype=int)
            for i_channel, channel_id in enumerate(self.__channel_ids):
                self.__correlation = self.__self_correlations[i_channel]
                self.__channel_pair = [channel_id, channel_id]
                self.__channel_positions = [self.__detector.get_relative_position(self.__station_id, channel_id),
                                            self.__detector.get_relative_position(self.__station_id, channel_id)]
                correlation_map = np.zeros_like(correlation_sum)
                for i_ray in range(len(self.__ray_types)):
                    if self.__ray_types[i_ray][0] != self.__ray_types[i_ray][1]:
                        self.__current_ray_types = self.__ray_types[i_ray]
                        correlation_map = np.maximum(self.get_correlation_array_3d(x_coords, y_coords, z_coords), correlation_map)
                self_correlation_sum += correlation_map
                number_of_dnr_channels += correlation_map.astype(bool)
            # set min number of channels / channel pairs to avoid division by 0 error
            number_of_channel_pairs = 1#[number_of_channel_pairs==0] = 1
            number_of_dnr_channels = .5#[number_of_dnr_channels==0] = 1
            combined_correlations = correlation_sum / number_of_channel_pairs + 0.5 * self_correlation_sum / number_of_dnr_channels

            mean_corr = np.nanmean(
                np.where(combined_correlations > 0, combined_correlations, np.nan),
                axis=0
            )
            max_corr = np.nanmax(combined_correlations, axis=0)
            mean_corr[np.isnan(mean_corr)] = 0
            i_max_dnr = np.unravel_index(np.argmax(combined_correlations), combined_correlations.shape)
            vertex_x = x_coords[i_max_dnr]
            vertex_y = y_coords[i_max_dnr]
            vertex_z = z_coords[i_max_dnr]
            fit_vertex = np.array([vertex_x, vertex_y, vertex_z])
            station.set_parameter(stnp.nu_vertex, fit_vertex)
            for sim_shower in event.get_sim_showers():
                sim_vertex = sim_shower.get_parameter(shp.vertex)
                fit_correlation = combined_correlations[i_max_dnr]
                # get correlation for true vertex position
                # sim_correlation = -self.__full_correlation_for_pos(sim_vertex)
                logger.debug(f"Sim vertex: ({sim_vertex[0]:.1f}, {sim_vertex[1]:.1f}, {sim_vertex[2]:.1f})")
                logger.debug(f"Fit vertex: ({vertex_x:.1f}, {vertex_y:.1f}, {vertex_z:.1f})")
                # logger.debug(f"Sim correlation: {sim_correlation:.3g}")
                logger.debug(f"Fit correlation: {fit_correlation:.3g}")
                max_pair_correlations = np.sum(np.max(self.__pair_correlations, axis=1))
                max_dnr_correlations = np.sum(np.max(self.__self_correlations, axis=1))
                logger.debug(f"Maximum correlation: {max_pair_correlations+max_dnr_correlations:.3g} ({max_pair_correlations:.3g} + {max_dnr_correlations:.3g} DnR)")
                # fit_correlation = -self.__full_correlation_for_pos(fit_vertex)
                # logger.debug(f"Fit correlation (?): {fit_correlation:.3g}")
                # if sim_correlation > 1.01 * fit_correlation:
                #     logger.warning(
                #         f"Correlation for simulated vertex ({sim_correlation:.3g}) higher than fit ({fit_correlation:.3g})!")
                break


            phi_fit = np.arctan2(vertex_y, vertex_x) % (2*np.pi)
            theta_fit = np.pi/2 - np.arctan2(vertex_z + 5 * units.m, np.linalg.norm([vertex_x, vertex_y]))
            distance_fit = np.linalg.norm([vertex_x, vertex_y, vertex_z]) # due to the 5 m offset, this is not entirely consistent - maybe fix? #TODO
            phi_sim = np.arctan2(sim_vertex[1], sim_vertex[0]) % (2*np.pi)
            theta_sim = np.pi/2 - np.arctan2(sim_vertex[2] + 5 * units.m, np.linalg.norm(sim_vertex[:2]))
            distance_sim = np.linalg.norm(sim_vertex)

            if debug:
                self.__draw_correlation_map_polar(
                    event, thetaphi, mean_corr,
                    thetaphi_sim = [theta_sim, phi_sim],
                    thetaphi_fit = [theta_fit, phi_fit],
                    filetag=f'_mean_{i_iteration}'
                )
                self.__draw_correlation_map_polar(
                    event, thetaphi, max_corr,
                    thetaphi_sim = [theta_sim, phi_sim],
                    thetaphi_fit = [theta_fit, phi_fit],
                    filetag=f'_max_{i_iteration}'
                )

                zoom_fit = (np.abs(thetaphi[0] - theta_fit) < 10 * dTheta) & (np.abs((thetaphi[1] - phi_fit + np.pi) % (2*np.pi) - np.pi) < 10 * dTheta)
                self.__draw_correlation_distance(
                    event, thetaphi[:,zoom_fit], distances_3d,
                    combined_correlations[:, zoom_fit],
                    fit=[distance_fit, theta_fit, phi_fit],
                    sim=[distance_sim, theta_sim, phi_sim],
                    filetag=f'_fit_{i_iteration}'
                )

                zoom_sim = (np.abs(thetaphi[0] - theta_sim) < 10 * dTheta) & (np.abs((thetaphi[1] - phi_sim + np.pi) % (2*np.pi) - np.pi) < 10 * dTheta)
                if not np.sum(zoom_sim):
                    logger.debug("Simulated vertex outside search area - skipping simulated vertex debug plot")
                else:
                    self.__draw_correlation_distance(
                        event, thetaphi[:,zoom_sim], distances_3d,
                        combined_correlations[:, zoom_sim],
                        fit=[distance_fit, theta_fit, phi_fit],
                        sim=[distance_sim, theta_sim, phi_sim],
                        filetag=f'_sim_{i_iteration}'
                    )

            mean_cutoff = np.percentile(mean_corr, 90)
            max_cutoff = np.percentile(max_corr, 95)
            if (i_iteration > 2) or (not self._use_maximum_filter):
                thetaphi_mask = thetaphi[:,np.where((mean_corr >= mean_cutoff) | (max_corr >= max_cutoff))[0]]
            else:
                thetaphi_mask = thetaphi[:, np.where(mean_corr >= mean_cutoff)[0]]

            dTheta /= 2
            n_distance_steps = int(n_distance_steps * np.sqrt(2))
            i_iteration += 1




        # logger.warning("starting scipy fitter...")
        # fitter_output = scipy.optimize.fmin(
        #     self.__full_correlation_for_pos, fit_vertex, full_output=True, xtol=1.)
        # #     minimizer_kwargs=dict(full_output=True))
        # logger.warning("...finished. Results:")
        # print(fitter_output)

        # dist_corrs = np.max(np.max(combined_correlations, axis=0), axis=1)
        # station.set_parameter(stnp.distance_correlations, dist_corrs)
        # station.set_parameter(stnp.vertex_search_path, [slope, offset, median_theta])
        # station.set_parameter(stnp.vertex_correlation_sums, [fit_correlation, sim_correlation, max_pair_correlations, max_dnr_correlations])
        if debug:
            # self.__draw_dnr_reco(
            #     event,
            #     correlation_sum / np.max(correlation_sum),
            #     self_correlation_sum / np.max(self_correlation_sum),
            #     combined_correlations / np.max(combined_correlations),
            #     x_0,
            #     y_0,
            #     z_0,
            #     slope,
            #     offset,
            #     median_theta,
            #     i_max,
            #     i_max_dnr
            # )
            # np.save(
            #     '{}/{}_{}_dnr_correlation.npy'.format(self.__debug_folder, event.get_run_number(), event.get_id()),
            #     self_correlation_sum
            # )
            fit_vx_times = dict()
            sim_vx_times = dict()
            for channel_id in self.__channel_ids:
                channel_pos = self.__detector.get_relative_position(self.__station_id, channel_id)
                # for fit vertex
                d_hor = np.atleast_2d(np.linalg.norm((fit_vertex-channel_pos)[:2]))
                z = np.atleast_2d(fit_vertex[2])
                fit_vx_times[channel_id] = dict()
                for ray_type in ['D', 'R']:#['direct', 'refracted', 'reflected']:
                    dt = self.get_signal_travel_time(
                        d_hor, z, ray_type, channel_id)[0,0]
                    if dt < 0.01 * units.ns:
                        dt = np.nan
                    fit_vx_times[channel_id][ray_type] = dt
                # for sim vertex
                d_hor = np.atleast_2d([np.linalg.norm((sim_vertex-channel_pos)[:2])])
                z = np.atleast_2d(sim_vertex[2])
                sim_vx_times[channel_id] = dict()
                for ray_type in ['D','R']:#['direct', 'refracted', 'reflected']:
                    dt = self.get_signal_travel_time(
                        d_hor, z, ray_type, channel_id)[0,0]
                    if dt < 0.01 * units.ns:
                        dt = np.nan
                    sim_vx_times[channel_id][ray_type] = dt
            self.__draw_pair_correlations(event, fit_vx_times, sim_vx_times)
            self.__draw_dnr_correlations(event, fit_vx_times, sim_vx_times)


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
        # logger.debug("2d correlation - array shape {}".format(str(d_hor1.shape)))
        res = self.get_correlation_for_pos(np.array([d_hor1, d_hor2]), z)
        return res

    def get_correlation_array_3d(self, x, y, z, d_r=0, d_z=0):
        channel_pos1 = self.__channel_positions[0]
        channel_pos2 = self.__channel_positions[1]
        d_hor1 = np.sqrt((x - channel_pos1[0])**2 + (y - channel_pos1[1])**2)
        d_hor2 = np.sqrt((x - channel_pos2[0])**2 + (y - channel_pos2[1])**2)
        # logger.debug("3d correlation - array shape {}".format(str(d_hor1.shape)))
        res = self.get_correlation_for_pos(np.array([d_hor1, d_hor2]), z)
        # res[np.abs(res) < .8 * np.max(np.abs(res))] = 0
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

        t_1 = self.get_signal_travel_time(d_hor[0], z, self.__current_ray_types[0], self.__channel_pair[0])
        t_2 = self.get_signal_travel_time(d_hor[1], z, self.__current_ray_types[1], self.__channel_pair[1])
        delta_t = t_1 - t_2
        delta_start_time = self.__start_times[self.__channel_pair[1]] - self.__start_times[self.__channel_pair[0]]
        delta_t = delta_t.astype(float)
        # print(d_hor[0].shape, z.shape, delta_t.shape)
        # print(t_1)
        # print(delta_t)
        # To avoid missing maxima because of our finite grid spacing,
        # we use a maximum filter. We determine the footprint of the filter
        # from the resolution of our grid in distance, theta and phi
        time_deviations = np.zeros_like(delta_t)
        mask_invalid = (delta_t[1:] == 0) | (delta_t[:-1] == 0)
        delta_t_offset = np.abs(delta_t[1:] - delta_t[:-1])
        delta_t_offset[mask_invalid] = 0
        # np.savez(f'debug_dt_{self.__current_ray_types[0]}_{self.__current_ray_types[1]}', dt=delta_t_offset, d_hor=d_hor,z=z, t_1=t_1,t_2=t_2)
        time_deviations[:-1] = delta_t_offset
        # dtdt = np.copy(time_deviations)
        # dtdt[dtdt==0] = np.nan
        # dtdt = np.nanmedian(dtdt, axis=0)
        # dtdt[np.isnan(dtdt)] = 0
        # self.__draw_correlation_map_polar(self.__event, self.__thetaphi, dtdt, percentile=10, filetag=f'_dt_distance_{self.__channel_pair}')
        delta_t_offset = np.abs(delta_t - delta_t[:, self.__nn_theta])
        mask_invalid = (delta_t == 0) | (delta_t[:, self.__nn_theta] == 0)
        delta_t_offset[mask_invalid] = 0
        time_deviations = np.maximum(time_deviations, delta_t_offset)
        # dtdt = np.copy(delta_t_offset)
        # dtdt[dtdt==0] = np.nan
        # dtdt = np.nanmedian(dtdt, axis=0)
        # dtdt[np.isnan(dtdt)] = 0
        # self.__draw_correlation_map_polar(self.__event, self.__thetaphi, dtdt, percentile=10, filetag=f'_dt_theta_{self.__channel_pair}')
        delta_t_offset = np.abs(delta_t - delta_t[:, self.__nn_phi])
        mask_invalid = (delta_t == 0) | (delta_t[:, self.__nn_phi] == 0)
        delta_t_offset[mask_invalid] = 0
        time_deviations = np.maximum(time_deviations, delta_t_offset)
        # dtdt = np.copy(delta_t_offset)
        # dtdt[dtdt==0] = np.nan
        # dtdt = np.nanmedian(dtdt, axis=0)
        # dtdt[np.isnan(dtdt)] = 0
        # self.__draw_correlation_map_polar(self.__event, self.__thetaphi, dtdt, percentile=10, filetag=f'_dt_phi_{self.__channel_pair}')

        time_deviations[np.isnan(time_deviations)] = 0

        corr_index = self.__correlation.shape[0] / 2 + np.round((delta_t + delta_start_time) * self.__sampling_rate)
        corr_index[np.isnan(delta_t)] = -1
        mask = (corr_index >= 0) & (corr_index < self.__correlation.shape[0]) & (~np.isinf(delta_t))
        corr_index[~mask] = 0
        # res = np.take(self.__correlation, corr_index.astype(int))
        # res[~mask] = 0
        # return res

        res = np.zeros_like(corr_index)
        if self._use_maximum_filter:
            step_size_x = int(np.ceil(res.shape[0] / 100))
            step_size_y = int(np.ceil(res.shape[1]))
            for i_x in range(res.shape[0] // step_size_x + 1):
                for i_y in range(res.shape[1] // step_size_y + 1):
                    i_x_0 = i_x * step_size_x
                    i_x_1 = i_x_0 + step_size_x
                    i_y_0 = i_y * step_size_y
                    i_y_1 = i_y_0 + step_size_y
                    time_deviations_i = time_deviations[i_x_0:i_x_1, i_y_0:i_y_1]
                    size = 0
                    if np.any(time_deviations_i > 0):
                        size = np.median(
                            time_deviations_i[time_deviations_i > 0]
                        ) * self.__sampling_rate #/ 2
                    maximized_correlation = scipy.ndimage.maximum_filter(
                                np.abs(self.__correlation),
                                size=size
                            )
                    res[i_x_0:i_x_1, i_y_0:i_y_1] = np.take(maximized_correlation, corr_index[i_x_0:i_x_1, i_y_0:i_y_1].astype(int))
        else:
            res = np.abs(self.__correlation)[corr_index.astype(int)]
        res[~mask] = 0
        res[np.isnan(res)] = 0
        # print(time_deviations)
        # print(res)
        """
        corr_index = self.__correlation.shape[0] / 2 + np.round(delta_t * self.__sampling_rate)
        corr_index[np.isnan(delta_t)] = 0
        mask = (corr_index > 0) & (corr_index < self.__correlation.shape[0]) & (~np.isinf(delta_t))
        corr_index[~mask] = 0
        res = np.take(self.__correlation, corr_index.astype(int))
        res[~mask] = 0
        """
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
        # logger.debug("get_signal_travel_time: array shape {}, ray type {}".format(d_hor.shape, ray_type))
        channel_pos = self.__detector.get_relative_position(self.__station_id, channel_id)
        channel_type = int(abs(channel_pos[2]))
        # travel_times = self.__get_time_scipy[channel_type][ray_type](d_hor, z)
        # return travel_times
        travel_times = np.zeros_like(d_hor)
        mask = np.ones_like(travel_times).astype(bool)
        i_z_1 = np.array(np.floor((z - self.__header[channel_type]['z_min']) / self.__header[channel_type]['d_z'])).astype(int)
        z_dist_1 = i_z_1 * self.__header[channel_type]['d_z'] + self.__header[channel_type]['z_min']
        i_z_2 = np.array(np.ceil((z - self.__header[channel_type]['z_min']) / self.__header[channel_type]['d_z'])).astype(int)
        z_dist_2 = i_z_2 * self.__header[channel_type]['d_z'] + self.__header[channel_type]['z_min']
        i_x_1 = np.array(np.floor((d_hor - self.__header[channel_type]['x_min']) / self.__header[channel_type]['d_x'])).astype(int)
        cell_dist_1 = i_x_1 * self.__header[channel_type]['d_x'] + self.__header[channel_type]['x_min']
        i_x_2 = np.array(np.ceil((d_hor - self.__header[channel_type]['x_min']) / self.__header[channel_type]['d_x'])).astype(int)
        cell_dist_2 = i_x_2 * self.__header[channel_type]['d_x'] + self.__header[channel_type]['x_min']
        mask[i_x_1 > self.__lookup_table[channel_type][ray_type].shape[0] - 1] = False
        mask[i_x_2 > self.__lookup_table[channel_type][ray_type].shape[0] - 1] = False
        mask[i_z_1 > self.__lookup_table[channel_type][ray_type].shape[1] - 1] = False
        mask[i_z_2 > self.__lookup_table[channel_type][ray_type].shape[1] - 1] = False
        for ind in [i_x_1, i_x_2, i_z_1, i_z_2]:
            mask[ind < 0] = False # make sure all indices are positive (=> not outside lookup table)
        i_x_1[~mask] = 0
        i_x_2[~mask] = 0
        i_z_1[~mask] = 0
        i_z_2[~mask] = 0
        travel_times_1_1 = self.__lookup_table[channel_type][ray_type][(i_x_1, i_z_1)]
        travel_times_1_2 = self.__lookup_table[channel_type][ray_type][(i_x_1, i_z_2)]
        travel_times_2_1 = self.__lookup_table[channel_type][ray_type][(i_x_2, i_z_1)]
        travel_times_2_2 = self.__lookup_table[channel_type][ray_type][(i_x_2, i_z_2)]
        valid_mask = (
            travel_times_1_1.astype(bool) & travel_times_1_2.astype(bool)
            & travel_times_2_1.astype(bool) & travel_times_2_2.astype(bool)
        )
        mask[~valid_mask] = False
        z_slopes_1 = np.zeros_like(travel_times_1_1)
        z_slopes_2 = np.zeros_like(travel_times_1_1)
        z_slopes_1[i_z_1 < i_z_2] = (travel_times_1_1 - travel_times_1_2)[i_z_1 < i_z_2] / (z_dist_1 - z_dist_2)[i_z_1 < i_z_2]
        z_slopes_2[i_z_1 < i_z_2] = (travel_times_2_1 - travel_times_2_2)[i_z_1 < i_z_2] / (z_dist_1 - z_dist_2)[i_z_1 < i_z_2]
        travel_times_1 = (z - z_dist_1) * z_slopes_1 + travel_times_1_1
        travel_times_2 = (z - z_dist_1) * z_slopes_2 + travel_times_2_1
        d_slope = np.zeros_like(z)
        d_slope[i_x_2 > i_x_1] = (travel_times_1 - travel_times_2)[i_x_2 > i_x_1] / (cell_dist_1 - cell_dist_2)[i_x_2 > i_x_1]
        travel_times = (d_hor - cell_dist_1) * d_slope + travel_times_1
        travel_times[~mask] = np.nan
        return travel_times

    def __full_correlation_for_pos(self, pos, spherical_cs=False):
        if spherical_cs:
            pos = pos[0] * np.array([np.sin(pos[1])*np.cos(pos[2]), np.sin(pos[1])*np.sin(pos[2]), np.cos(pos[1])])
        x_coords, y_coords, z_coords = np.meshgrid(*pos)
        correlation_sum_sim = np.zeros_like(z_coords)
        # channel-channel
        for i_pair, channel_pair in enumerate(self.__channel_pairs):
            self.__channel_pair = channel_pair
            self.__correlation = self.__pair_correlations[i_pair]
            self.__channel_positions = [self.__detector.get_relative_position(self.__station_id, channel_pair[0]),
                                        self.__detector.get_relative_position(self.__station_id, channel_pair[1])]
            correlation_map_sim = np.zeros_like(correlation_sum_sim)
            for i_ray in range(len(self.__ray_types)):
                self.__current_ray_types = self.__ray_types[i_ray]
                corr = self.get_correlation_array_3d(x_coords, y_coords, z_coords)
                # print(channel_pair, i_ray, corr)
                correlation_map_sim = np.maximum(corr, correlation_map_sim)
            # print(channel_pair, correlation_map_sim)
            correlation_sum_sim += correlation_map_sim
        # DnR
        self_correlation_sum_sim = np.zeros_like(z_coords)
        for i_channel, channel_id in enumerate(self.__channel_ids):
            self.__channel_pair = [channel_id, channel_id]
            self.__correlation = self.__self_correlations[i_channel]
            self.__channel_positions = [self.__detector.get_relative_position(self.__station_id, channel_id),
                                        self.__detector.get_relative_position(self.__station_id, channel_id)]
            correlation_map_sim = np.zeros_like(correlation_sum_sim)
            for i_ray in range(len(self.__ray_types)):
                if self.__ray_types[i_ray][0] != self.__ray_types[i_ray][1]:
                    self.__current_ray_types = self.__ray_types[i_ray]
                    correlation_map_sim = np.maximum(self.get_correlation_array_3d(x_coords, y_coords, z_coords), correlation_map_sim)
            self_correlation_sum_sim += correlation_map_sim
        combined_correlations_sim = correlation_sum_sim + self_correlation_sum_sim
        return -np.max(combined_correlations_sim)

    def __draw_correlation_map_polar(self, event, thetaphi, correlation, thetaphi_fit=None, thetaphi_sim=None, percentile=90, filetag=''):
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(projection='polar')
        markerSize = 5*self._dTheta_current/units.deg

        phi_mw = thetaphi[1]
        cutoff = np.percentile(correlation, percentile)
        y= thetaphi[0] / units.deg
        mask = correlation >= cutoff
        ax.plot(phi_mw[~mask], y[~mask], '.', ms=markerSize/10, color='gray', alpha=.5,zorder=0)
        cax = ax.scatter(phi_mw[mask], y[mask], marker='h', s=markerSize, c=correlation[mask])
        cmap = cax.get_cmap()
        bg = list(cmap(0))
        ax.set_facecolor(bg)

        if thetaphi_fit is not None:
            plt.plot(thetaphi_fit[1], thetaphi_fit[0] / units.deg, marker='s', mfc='none', color='lightblue', ms=10, label='fit', ls='')
        if thetaphi_sim is not None:
            plt.plot(thetaphi_sim[1], thetaphi_sim[0] / units.deg, marker='o', mfc='none', color='red', ms=10, label='sim', ls='')

        plt.ylim(180,90)
        yticks = np.arange(165, 89, -15)
        ticklabels=[f'{y:.0f}$^\circ$' for y in yticks]
        ax.set_yticks(yticks,labels=ticklabels, color='white')
        ax.get_yticklabels()[-1].set_color('black')
        ax.grid(color='gray')
        if (thetaphi_fit is not None) or (thetaphi_sim is not None):
            ax.legend()
        plt.colorbar(cax)
        plt.tight_layout()
        fname = '{}/{}_{}_correlation_polar{}'.format(self.__debug_folder, event.get_run_number(), event.get_id(), filetag)
        save_fig(fig, fname, self.__debug_fmts)
        plt.close()

    def __draw_correlation_distance(self, event, thetaphi, distances, correlation, fit=None, sim=None, filetag=''):
        fig = plt.figure(figsize=(12,4))
        # distance vs zenith
        ax = fig.add_subplot(1,2,1)
        vmin = np.min(correlation[correlation>0])
        # we create the arange 'backwards' because the maximum value needs to be included for correct sorting
        thetamin = np.min(thetaphi[0])
        thetamax = np.max(thetaphi[0])
        n_theta = int(np.round((thetamax - thetamin) / self._dTheta_current)) + 1
        theta = np.linspace(thetamin, thetamax, n_theta)
        # theta = np.arange(np.max(thetaphi[0]), np.min(thetaphi[0]) - self._dTheta_current/10, -self._dTheta_current)[::-1]
        corr_grid = np.zeros((len(distances), len(theta)))
        theta_index = np.searchsorted(theta, thetaphi[0])
        # print(thetaphi.shape, correlation.shape)
        # print(theta.shape, theta_index.shape)
        # print(np.max(thetaphi[0]/units.deg), thetaphi[0] / units.deg)
        # print(np.max(theta/units.deg), theta / units.deg)
        # print(np.max(theta_index), theta_index)
        for i, i_theta in enumerate(theta_index):
            corr_grid[:, i_theta] = np.maximum(corr_grid[:, i_theta], correlation[:, i])
        cmap = plt.cm.get_cmap('viridis').copy()
        cmap.set_under('white')
        cax = ax.pcolormesh(distances, theta/units.deg, corr_grid.T, vmin=vmin, cmap=cmap)
        ax.set_xscale('log')
        ylims = ax.get_ylim()

        if not fit is None:
            ax.plot(fit[0], fit[1]/units.deg, marker='s', mfc='none', color='black', ms=10, label='fit', ls='')
        if not sim is None:
            ax.plot(sim[0], sim[1]/units.deg, marker='o', mfc='none', color='red', ms=10, label='sim', ls='')
        ax.set_ylim(ylims[1], ylims[0])
        ax.set_xlabel('distance [m]')
        ax.set_ylabel('zenith [deg]')

        # distance vs azimuth
        ax = fig.add_subplot(1,2,2)
        if np.max(thetaphi[1]) - np.min(thetaphi[1]) > np.pi: # discontinuity -> shift to [-pi, pi]
            phi = (thetaphi[1] + np.pi) % (2 * np.pi) - np.pi
            if fit:
                fit[2] = (fit[2] + np.pi) % (2 * np.pi) - np.pi
            if sim:
                sim[2] = (sim[2] + np.pi) % (2 * np.pi) - np.pi
        else:
            phi = thetaphi[1]
        thetamin = np.min(phi)
        thetamax = np.max(phi)
        n_theta = int(np.round((thetamax - thetamin) / self._dTheta_current)) + 1
        theta = np.linspace(thetamin, thetamax, n_theta)
        # theta = np.arange(np.max(phi), np.min(phi)-self._dTheta_current, -self._dTheta_current)[::-1]
        corr_grid = np.zeros((len(distances), len(theta)))
        theta_index = np.searchsorted(theta, phi)
        for i, i_theta in enumerate(theta_index):
            corr_grid[:, i_theta] = np.maximum(corr_grid[:, i_theta], correlation[:, i])
        cax = ax.pcolormesh(distances, theta/units.deg, corr_grid.T, vmin=vmin, cmap=cmap)
        ax.set_xscale('log')
        ylims = ax.get_ylim()

        if not fit is None:
            logger.debug(f"Fit: ({fit[0]:.0f}, {fit[1]/units.deg:.2f}, {fit[2]/units.deg:.2f})")
            ax.plot(fit[0], fit[2]/units.deg, marker='s', mfc='none', color='black', ms=10, label='fit', ls='')
        if not sim is None:
            logger.debug(f"Sim: ({sim[0]:.0f}, {sim[1]/units.deg:.2f}, {sim[2]/units.deg:.2f})")
            ax.plot(sim[0], sim[2]/units.deg, marker='o', mfc='none', color='red', ms=10, label='sim', ls='')
        ax.set_ylim(ylims)
        ax.set_xlabel('distance [m]')
        ax.set_ylabel('azimuth [deg]')

        ax.legend()
        plt.colorbar(cax)
        plt.tight_layout()
        fname = '{}/{}_{}_correlation_zoom{}'.format(self.__debug_folder, event.get_run_number(), event.get_id(), filetag)
        save_fig(fig, fname, self.__debug_fmts)
        plt.close()


    def __draw_pair_correlations(self, event,  fit_times=dict(), sim_times=dict()):
        fig1 = plt.figure(figsize=(12, (len(self.__channel_pairs) + len(self.__channel_pairs) % 2)))
        for i_pair, channel_pair in enumerate(self.__channel_pairs):
            channel_id1, channel_id2 = channel_pair
            correlation_product = self.__pair_correlations[i_pair]
            toffset = np.arange(-len(correlation_product) // 2, len(correlation_product) // 2, dtype=int)
            toffset = toffset / self.__sampling_rate
            ax1_1 = fig1.add_subplot(len(self.__channel_pairs) // 2 + len(self.__channel_pairs) % 2, 2,
                                        i_pair + 1)
            # plot time offsets from fit / simulation
            rt_dict = dict(direct=0,refracted=1,reflected=2,D='D',R='R')
            ymax = np.max(correlation_product)
            if (channel_id1 in fit_times) & (channel_id2 in fit_times):
                for ray_types in self.__ray_types:
                    dt = fit_times[channel_id1][ray_types[0]] - fit_times[channel_id2][ray_types[1]]
                    dt -= self.__start_times[channel_id1] - self.__start_times[channel_id2]
                    ax1_1.axvline(dt, ls=':', color='black')
                    if (dt < toffset[-1]) & (dt > toffset[0]):
                        ax1_1.text(dt, .95*ymax, f'{rt_dict[ray_types[0]]}&{rt_dict[ray_types[1]]}', ha='center')
            if (channel_id1 in sim_times) & (channel_id2 in sim_times):
                for ray_types in self.__ray_types:
                    dt = sim_times[channel_id1][ray_types[0]] - sim_times[channel_id2][ray_types[1]]
                    dt -= self.__start_times[channel_id1] - self.__start_times[channel_id2]
                    ax1_1.axvline(dt, ls=':', color='red', alpha=.6)
                    if (dt < toffset[-1]) & (dt > toffset[0]):
                        ax1_1.text(dt, .85*ymax, f'{rt_dict[ray_types[0]]}&{rt_dict[ray_types[1]]}', color='red', ha='center')

            ax1_1.grid()
            ax1_1.plot(toffset, correlation_product)
            ax1_1.set_xlim(1.2*toffset[0], 1.2*toffset[-1])
            ax1_1.set_title('Ch.{} & Ch.{}'.format(channel_pair[0], channel_pair[1]))
            ax1_1.set_xlabel(r'$\Delta t$ [ns]')
            ax1_1.set_ylabel('correlation')
        fig1.tight_layout()
        fname = '{}/{}_{}_correlation'.format(self.__debug_folder, event.get_run_number(), event.get_id())
        save_fig(fig1, fname, self.__debug_fmts)
        plt.close()

    def __draw_dnr_correlations(self, event, fit_times=dict(), sim_times=dict()):
        fig1 = plt.figure(figsize=(12, (len(self.__channel_ids) + len(self.__channel_ids) % 2)))
        for i_ch, channel_id in enumerate(self.__channel_ids):
            # channel_id1, channel_id2 = channel_pair
            correlation_product = self.__self_correlations[i_ch]
            toffset = np.arange(-len(correlation_product) // 2, len(correlation_product) // 2, dtype=int)
            toffset = toffset / self.__sampling_rate
            ax1_1 = fig1.add_subplot(len(self.__channel_ids) // 2 + len(self.__channel_ids) % 2, 2,
                                        i_ch + 1)
            # plot time offsets from fit / simulation
            rt_dict = dict(direct=0,refracted=1,reflected=2,D='D',R='R')
            ymax = np.max(correlation_product)
            if (channel_id in fit_times):
                for ray_types in self.__ray_types:
                    if ray_types[0] != ray_types[1]:
                        dt = fit_times[channel_id][ray_types[0]] - fit_times[channel_id][ray_types[1]]
                        ax1_1.axvline(dt, ls=':', color='black')
                        if (dt < toffset[-1]) & (dt > toffset[0]):
                            ax1_1.text(dt, .95*ymax, f'{rt_dict[ray_types[0]]}&{rt_dict[ray_types[1]]}', ha='center')
            if (channel_id in sim_times):
                for ray_types in self.__ray_types:
                    if ray_types[0] != ray_types[1]:
                        dt = sim_times[channel_id][ray_types[0]] - sim_times[channel_id][ray_types[1]]
                        ax1_1.axvline(dt, ls=':', color='red', alpha=.6)
                        if (dt < toffset[-1]) & (dt > toffset[0]):
                            ax1_1.text(dt, .85*ymax, f'{rt_dict[ray_types[0]]}&{rt_dict[ray_types[1]]}', color='red', ha='center')

            ax1_1.grid()
            ax1_1.plot(toffset, correlation_product)
            ax1_1.set_xlim(1.2*toffset[0], 1.2*toffset[-1])
            ax1_1.set_title(f'Ch.{channel_id}')
            ax1_1.set_xlabel(r'$\Delta t$ [ns]')
            ax1_1.set_ylabel('correlation')
        fig1.tight_layout()
        fname = '{}/{}_{}_dnr_correlation'.format(self.__debug_folder, event.get_run_number(), event.get_id())
        save_fig(fig1, fname, self.__debug_fmts)
        plt.close()

    def __draw_2d_correlation_map(self, event, correlation_map, slope, offset, max_z_offset, min_z_offset):
        fig4 = plt.figure(figsize=(6, 3.5))
        ax4_1 = fig4.add_subplot(111)
        d_0, z_0 = np.meshgrid(self.__distances_2d, self.__z_coordinates_2d)
        cplot = ax4_1.pcolor(
            d_0,
            z_0,
            np.max(correlation_map, axis=2).T / np.max(correlation_map),
            vmin=.5,
            vmax=1,
            shading='auto'
        )
        plt.colorbar(cplot, ax=ax4_1).set_label('correlation sum', fontsize=14)
        ax4_1.plot(
            self.__distances_2d,
            self.__distances_2d * slope + offset + max_z_offset,
            color='k',
            alpha=1.
        )
        ax4_1.plot(
            self.__distances_2d,
            self.__distances_2d * slope + offset - min_z_offset,
            color='k',
            alpha=1.
        )
        ax4_1.fill_between(
            self.__distances_2d,
            self.__distances_2d * slope + offset + max_z_offset,
            self.__distances_2d * slope + offset - min_z_offset,
            color='k',
            alpha=.2
        )
        ax4_1.set_xlim([np.min(self.__distances_2d), np.max(self.__distances_2d)])
        ax4_1.set_ylim([np.min(self.__z_coordinates_2d), np.max(self.__z_coordinates_2d)])
        ax4_1.set_aspect('equal')
        ax4_1.grid()
        sim_vertex = None
        for sim_shower in event.get_sim_showers():
            sim_vertex = sim_shower.get_parameter(shp.vertex)
            break
        ax4_1.scatter(
            [np.sqrt(sim_vertex[0] ** 2 + sim_vertex[1] ** 2)],
            [sim_vertex[2]],
            c='r',
            alpha=1.,
            marker='o',
            s=50
        )
        ax4_1.set_xlabel('d [m]', fontsize=14)
        ax4_1.set_ylabel('z [m]', fontsize=14)
        fig4.tight_layout()
        fname = '{}/{}_{}_2D_correlation_maps'.format(self.__debug_folder, event.get_run_number(), event.get_id())
        save_fig(fig4, fname, self.__debug_fmts)

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
            median_theta,
            full_correlations
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

        ax5_2 = fig5.add_subplot(2, 2, 3)
        ax5_2.grid()
        ax5_2.plot(
            self.__azimuths_2d / units.deg,
            np.sum(np.max(full_correlations, axis=0), axis=0)
        )
        ax5_3 = fig5.add_subplot(2, 2, 4)
        ax5_3.grid()
        ax5_3.plot(
            self.__azimuths_2d / units.deg,
            np.max(np.sum(full_correlations, axis=0), axis=0)
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
            ax5_2.axvline(
                hp.get_normalized_angle(hp.cartesian_to_spherical(sim_vertex[0], sim_vertex[1], sim_vertex[2])[1]) / units.deg,
                color='r',
                alpha=.5,
                linestyle='--'
            )
            ax5_3.axvline(
                hp.get_normalized_angle(hp.cartesian_to_spherical(sim_vertex[0], sim_vertex[1], sim_vertex[2])[1]) / units.deg,
                color='r',
                alpha=.5,
                linestyle='--'
            )
        ax5_2.axvline(
            median_theta / units.deg,
            color='k'
        )
        ax5_3.axvline(
            median_theta / units.deg,
            color='k'
        )
        ax5_1_1.set_xlabel('d [m]')
        ax5_1_2.set_xlabel('d [m]')
        ax5_1_1.set_ylabel('z [m]')
        ax5_1_2.set_ylabel('z [m]')
        ax5_2.set_xlabel(r'$\phi [^\circ]$')
        ax5_3.set_xlabel(r'$\phi [^\circ]$')
        ax5_2.set_ylabel(r'$\Sigma_z\mathrm{max}_d q$')
        ax5_3.set_ylabel(r'$\mathrm{max}_z\Sigma_d q$')
        if z_fit:
            ax5_3.set_facecolor('lightgrey')
        else:
            ax5_2.set_facecolor('lightgrey')
        #     ax5_2.set_xlabel(r'$\phi [^\circ]$')
        #     ax5_2.set_ylabel('|z| [m]')
        fig5.tight_layout()
        fname = '{}/{}_{}_search_zones'.format(self.__debug_folder, event.get_run_number(), event.get_id())
        save_fig(fig5, fname, self.__debug_fmts)

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
        fig6 = plt.figure(figsize=(6, 6))
        ax6_1 = fig6.add_subplot(222)
        vmin = .5
        vmax = 1.
        colormap = cm.get_cmap('viridis')
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
            vmax=vmax,
            shading='auto'
        )
        ax6_2 = fig6.add_subplot(224)
        cplot2 = ax6_2.pcolor(
            x_0[:, :, 0],
            y_0[:, :, 0],
            np.max(correlation_sum, axis=2),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )

        ax6_3 = fig6.add_subplot(221)
        cplot3 = ax6_3.pcolor(
            x_0[0],
            z_coords[0],
            np.max(correlation_sum, axis=0),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
        ax6_3.set_xlim([np.min(self.__distances_2d), np.max(self.__distances_2d)])
        ax6_3.set_ylim([np.min(self.__z_coordinates_2d), np.max(self.__z_coordinates_2d)])
        ax6_3.set_aspect('equal')
        ax6_4 = fig6.add_subplot(223)
        cplot4 = ax6_4.pcolor(
            x_coords[:, :, 0],
            y_coords[:, :, 0],
            np.max(correlation_sum, axis=2),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
        ax6_4.set_aspect('equal')
        ax6_4.set_xlim([-np.max(self.__distances_2d), np.max(self.__distances_2d)])
        ax6_4.set_ylim([-np.max(self.__distances_2d), np.max(self.__distances_2d)])
        if sim_vertex is not None:
            sim_vertex_dhor = np.sqrt(sim_vertex[0] ** 2 + sim_vertex[1] ** 2)
            ax6_1.scatter(
                [sim_vertex_dhor],
                [sim_vertex[2] - sim_vertex_dhor * slope - offset],
                c='r',
                marker='o',
                s=20, label='simulated'
            )
            ax6_2.scatter(
                [np.cos(median_theta) * sim_vertex[0] + np.sin(median_theta) * sim_vertex[1]],
                [-np.sin(median_theta) * sim_vertex[0] + np.cos(median_theta) * sim_vertex[1]],
                c='r',
                marker='o',
                s=20
            )
            ax6_3.scatter(
                [sim_vertex_dhor],
                [sim_vertex[2]],
                c='r',
                marker='o',
                s=20
            )
            ax6_4.scatter(
                [sim_vertex[0]],
                [sim_vertex[1]],
                c='r',
                marker='o',
                s=20
            )
        ax6_1.scatter(
            [x_0[i_max]],
            [z_0[i_max]],
            c='k',
            marker='x',
            s=20, label='fit'
        )
        ax6_2.scatter(
            [x_0[i_max]],
            [y_0[i_max]],
            c='k',
            marker='x',
            s=20
        )
        ax6_3.scatter(
            [x_0[i_max]],
            [z_coords[i_max]],
            c='k',
            marker='x',
            s=20
        )
        ax6_4.scatter(
            [x_coords[i_max]],
            [y_coords[i_max]],
            c='k',
            marker='x',
            s=20
        )
        fontsize = 12
        ax6_1.set_xlabel('r [m]', fontsize=fontsize)
        ax6_1.set_ylabel(r'$\Delta z$ [m]', fontsize=fontsize)
        ax6_2.set_xlabel('r [m]', fontsize=fontsize)
        ax6_2.set_ylabel(r'$\Delta y$ [m]', fontsize=fontsize)
        ax6_3.set_xlabel('r [m]', fontsize=fontsize)
        ax6_3.set_ylabel('z [m]', fontsize=fontsize)
        ax6_4.set_xlabel('x [m]', fontsize=fontsize)
        ax6_4.set_ylabel('y [m]', fontsize=fontsize)
        ax6_1.grid()
        ax6_2.grid()
        ax6_3.grid()
        ax6_4.grid()
        ax6_1.legend()
        fig6.tight_layout()
        fname = '{}/{}_{}_slices'.format(self.__debug_folder, event.get_run_number(), event.get_id())
        save_fig(fig6, fname, self.__debug_fmts)

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
        fig8 = plt.figure(figsize=(6, 8))
        ax8_1 = fig8.add_subplot(312)
        # ax8_2 = fig8.add_subplot(235)
        ax8_3 = fig8.add_subplot(313)
        # ax8_4 = fig8.add_subplot(236)
        ax8_5 = fig8.add_subplot(311)
        # ax8_6 = fig8.add_subplot(234)
        colormap = cm.get_cmap('viridis')
        fontsize = 14
        vmin = .5
        vmax = 1.
        cplot1 = ax8_1.pcolor(
            x_0[0],
            z_0[0],
            np.max(self_correlation_sum, axis=0),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
        plt.colorbar(cplot1, ax=ax8_1).set_label('correlation sum', fontsize=fontsize)
        cplot3 = ax8_3.pcolor(
            x_0[0],
            z_0[0],
            np.max(combined_correlations, axis=0),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
        plt.colorbar(cplot3, ax=ax8_3).set_label('correlation sum', fontsize=fontsize)
        cplot5 = ax8_5.pcolor(
            x_0[0],
            z_0[0],
            np.max(correlation_sum, axis=0),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
        plt.colorbar(cplot5, ax=ax8_5).set_label('correlation sum', fontsize=fontsize)
        ax8_5.scatter(
            [x_0[i_max]],
            [z_0[i_max]],
            c='k',
            marker='o',
            s=20
        )
        ax8_3.scatter(
            [x_0[i_max_dnr]],
            [z_0[i_max_dnr]],
            c='k',
            marker='o',
            s=20
        )
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
                marker='o',
                s=20
            )
            ax8_3.scatter(
                [sim_vertex_dhor],
                [sim_vertex[2] - sim_vertex_dhor * slope - offset],
                c='r',
                marker='o',
                s=20
            )
            ax8_5.scatter(
                [sim_vertex_dhor],
                [sim_vertex[2] - sim_vertex_dhor * slope - offset],
                c='r',
                marker='o',
                s=20
            )
        ax8_1.grid()
        ax8_3.grid()
        ax8_5.grid()
        ax8_5.set_title('channel correlation', fontsize=fontsize)
        ax8_1.set_title('DnR correlation', fontsize=fontsize)
        ax8_3.set_title('channel + DnR correlation', fontsize=fontsize)
        ax8_1.set_xlabel('r [m]', fontsize=fontsize)
        ax8_1.set_ylabel(r'$\Delta z$ [m]', fontsize=fontsize)
        ax8_3.set_xlabel('r [m]', fontsize=fontsize)
        ax8_3.set_ylabel(r'$\Delta z$ [m]', fontsize=fontsize)
        ax8_5.set_xlabel('r [m]', fontsize=fontsize)
        ax8_5.set_ylabel(r'$\Delta z$ [m]', fontsize=fontsize)
        # ax8_2.grid()
        fig8.tight_layout()
        fname = '{}/{}_{}_dnr_reco'.format(self.__debug_folder, event.get_run_number(), event.get_id())
        save_fig(fig8, fname, self.__debug_fmts)


def save_fig(fig, fname, format='.png'):
    """
    Save a matplotlib Figure instance

    Parameters
    ----------
    fig : matplotlib Fig instance
    fname : string
        location / name
    format : string | list (default: '.png')
        format(s) to save to save the figure to.
        If a list, save the figure to multiple formats.
        Can also include '.pickle'/'.pkl' to enable the Fig to be
        imported and edited in the future

    """
    formats = np.atleast_1d(format)
    for fmt in formats:
        if ('pickle' in fmt) or ('pkl' in fmt):
            with open(fname+'.pkl', 'wb') as file:
                pickle.dump(fig, file)
        else:
            if not fmt[0] == '.':
                fmt = '.' + fmt
            fig.savefig(fname+fmt)