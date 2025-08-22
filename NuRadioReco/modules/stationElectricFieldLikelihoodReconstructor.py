import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
import sys

from NuRadioReco.modules.base.module import register_run

import time
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy as scp

from NuRadioReco.utilities.analytic_pulse import get_analytic_pulse_freq
from NuRadioReco.utilities import units, noise_model, fft
from NuRadioReco.framework.electric_field import ElectricField
from NuRadioReco.framework.sim_station import SimStation
from NuRadioMC.simulation.simulation import apply_det_response_sim
from NuRadioReco.modules.efieldToVoltageConverter import efieldToVoltageConverter
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.framework.event
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.utilities.trace_utilities import get_electric_field_energy_fluence
from NuRadioReco.utilities import geometryUtilities as geo_utl
import NuRadioReco.modules.channelLengthAdjuster

efieldToVoltageConverter = efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False, pre_pulse_time=0, post_pulse_time=0, caching=False)
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
channelLengthAdjuster = NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()

from NuRadioReco.utilities.matched_filter import matched_filter, matched_filter_to_llh
from NuRadioReco.utilities.trace_minimizer import TraceMinimizer

import logging
logger = logging.getLogger('NuRadioReco.stationElectricFieldReconstructor')


class StationElectricFieldReconstructor:
    """
    Class for reconstructing electric field  from a dual-polarized antenna (two cannels)

    Parameters
    ----------
       detector: NuRadioReco.detector.detector.Detector
              Detector description of the detector that shall be simulated containing one station
        
        signal_model: NuRadioReco.framework.signal_model.SignalModel
                Signal model that describes the electric field signal
        
        noise_model: NuRadioReco.framework.noise_model.NoiseModel
                Noise model that describes the electric field noise
    """

    def __init__(self):
        pass

    def begin(self, n_channels, n_samples, sampling_rate, noise_spectrum, Vrms, filter_settings_low=None, filter_settings_high=None, use_chi2=False, zenith_azimuth_free=False):

        self.noise_spectrum = noise_spectrum
        self.Vrms = Vrms
        self.filter_settings_low = filter_settings_low
        self.filter_settings_high = filter_settings_high
        #self.use_channels = use_channels if use_channels is not None else detector.get_channel_ids(detector.get_station_ids()[0])
        self.zenith_azimuth_free = zenith_azimuth_free

        self.n_channels = n_channels
        self.n_samples = n_samples #detector.get_number_of_samples(self.station_id, self.channel_ids[0])
        self.sampling_rate = sampling_rate #detector.get_sampling_frequency(self.station_id, self.channel_ids[0])
        self.delta_t = 1/self.sampling_rate
        self.t_array = np.arange(0, self.n_samples) * self.delta_t
        self.t_array_matched_filter = np.copy(self.t_array) - self.t_array[int(self.n_samples/2)]
        self.i_shift_cc = np.arange(0, self.n_samples)
        self.frequencies = np.fft.rfftfreq(self.n_samples, 1. / self.sampling_rate)
        
        # initialize noise model:
        self.noise_model = noise_model.NoiseModel(n_antennas=self.n_channels, n_samples=self.n_samples, sampling_rate=self.sampling_rate, matrix_inversion_method="pseudo_inv", threshold_amplitude=0.1)
        self.noise_model.initialize_with_spectra(self.noise_spectrum, self.Vrms)
        self.noise_psd = self.noise_model.noise_psd

        self.use_chi2 = use_chi2

    
    @register_run()
    def run(self, evt, station, det, use_channels=None, signal_search_window=None, use_MC_direction=False):

        if use_channels is None:
            use_channels = station.get_channel_ids()

        if use_MC_direction and (station.get_sim_station() is not None):
            zenith = station.get_sim_station()[stnp.zenith]
            azimuth = station.get_sim_station()[stnp.azimuth]
            sim_present = True
        else:
            logger.warning("Using reconstructed angles as no simulation present")
            zenith = station[stnp.zenith]
            azimuth = station[stnp.azimuth]
            sim_present = False

        traces = []
        for channel in station.iter_channels():
            if channel.get_id() not in use_channels:
                continue
            trace = channel.get_trace()
            traces.append(trace)
        traces = np.array(traces)

        f_theta_f_phi_initial = [[0.5, 1], [0.5, -1], [-0.5, 1], [-0.5, -1]]

        sid = station.get_id()

        def signal_function(parameters):
            return self.get_signal(parameters, det, sid, use_channels)

        llh_best = None
        for i in range(4):

            parameters_initial = np.array([f_theta_f_phi_initial[i][0], f_theta_f_phi_initial[i][1], -1, np.pi/2, 300, -10, zenith, azimuth])

            # try:
            llh, polarization_reco_llh_1, fluence_reco_llh_1, fitted_params, errors_fit, error_polarization, error_fluence = self.reconstruct_signal(
                traces, signal_function, parameters_initial, use_channels, second_order=True, signal_search_window=signal_search_window)
            # except Exception as e:
            #     continue
            
            print("LLH:", llh_best, llh)
            if llh_best is None or llh < llh_best:
                llh_best = np.copy(llh)
                fitted_params_llh_best = np.copy(fitted_params)
                polarization_reco_llh_best = np.copy(polarization_reco_llh_1)
                fluence_reco_llh_best = np.copy(fluence_reco_llh_1)
                errors_fit_best = np.copy(errors_fit)
                error_polarization_best = np.copy(error_polarization)
                error_fluence_best = np.copy(error_fluence)

        initial_signal = signal_function(parameters_initial)
        fitted_signal = signal_function(fitted_params_llh_best)

        A_theta = np.sign(fitted_params_llh_best[0])*np.abs(fitted_params_llh_best[0])**0.5
        A_phi = np.sign(fitted_params_llh_best[1])*np.abs(fitted_params_llh_best[1])**0.5
        polarization_reco = np.arctan2(A_phi, A_theta) / units.deg

        results_method = dict(
            method_name="LLH",
            polarization_reco=polarization_reco, #polarization_reco_llh_best / units.deg,
            fluence_reco=fluence_reco_llh_best,
            f_theta_reco=np.abs(fitted_params_llh_best[0]),
            f_phi_reco=np.abs(fitted_params_llh_best[1]),
            llh=llh_best,
            error_fit=errors_fit_best,
            error_polarization=error_polarization_best / units.deg,
            error_fluence=error_fluence_best,
            params=fitted_params_llh_best,
            signal_search_window=signal_search_window,
            n_dof = np.sum(np.linalg.matrix_rank(self.noise_model.cov_inv)),
            A_theta=A_theta,
            A_phi=A_phi,
            zenith_reco=fitted_params_llh_best[6] if self.zenith_azimuth_free else None,
            azimuth_reco=fitted_params_llh_best[7] if self.zenith_azimuth_free else None,
        )

        return results_method, fitted_signal, initial_signal

    def function_to_minimize_1(self, data, signal, plot=False):


        if not self.use_chi2:
            # Matched filter:
            t, x = matched_filter(data, signal, self.noise_psd, self.t_array_matched_filter, self.frequencies, self.n_channels, plot=plot) #[600:700:]

            # Convert to llh:
            #noise_model.calculate_minus_two_delta_llh(signal_with_noise, signal_true)
            minus_two_delta_llh = - matched_filter_to_llh(x, data, signal, self.noise_psd, self.frequencies, self.n_channels)
            
            #print(minus_two_delta_llh)

            if plot:

                s = np.fft.rfft(data, axis=1)
                h = np.fft.rfft(signal, axis=1)

                plt.figure(figsize=[20,3])
                plt.plot(self.frequencies,self.noise_psd[0])
                #plt.plot(self.frequencies,self.noise_psd[1], "--")
                plt.plot(self.frequencies, abs(s[0,:])/np.max(abs(s[0,:]))*np.max(self.noise_psd[0])*2, label="data")
                plt.plot(self.frequencies, abs(h[0,:])/np.max(abs(h[0,:]))*np.max(self.noise_psd[0])*2, "--", label="signal")
                plt.savefig("noise_psd.png")
                plt.close()
                
            #print("matched filter:", minus_two_delta_llh)
            return minus_two_delta_llh
        
        elif self.use_chi2:
            i_max, cross = self.cross_correlation(data, signal, shift_array=self.i_shift_cc, plot=plot)
            return -cross

    def function_to_minimize_2(self, data, signal, plot=False):
        """
        Calculate the log-likelihood
        """
        if not self.use_chi2:
            llh =  self.noise_model.calculate_minus_two_delta_llh(data, signal)
            #print("llh:", llh)
            return llh
        elif self.use_chi2:
            return self.chi2(data, signal)


    def cross_correlation(self, data, signal, shift_array, plot=False):
        """
        Calculate the cross-correlation between the data and the signal.

        Parameters
        ----------
        data: np.ndarray
            Data from the two antennas

        signal: np.ndarray
            Signal from the two antennas

        Returns
        -------
        float
            Cross-correlation between the data and the signal
        """

        #shift_array = np.where(np.isin(self.t_array, t_array_cc))[0]
        cross_correlation_array = np.zeros(len(shift_array))
        for i, shift in enumerate(shift_array):
            cross_correlation_array[i] = np.sum(data[0,:] * np.roll(signal[0,:], shift)) + np.sum(data[1,:] * np.roll(signal[1,:], shift)) / np.sqrt(np.sum(data[0,:]**2) * np.sum(signal[0,:]**2) + np.sum(data[1,:]**2) * np.sum(signal[1,:]**2))
            # if i == 0:
            #     print(i)
            #     print(self.t_array[i])
            #     plt.figure(figsize=[20,3])
            #     plt.plot(data[0,:], label="data 1")
            #     plt.plot(np.roll(signal[0,:], i), "--", label="signal 1")
            #     plt.plot(data[1,:], label="data 2")
            #     plt.plot(np.roll(signal[1,:], i), "--", label="signal 2")
            #     plt.legend()
            #     plt.savefig("cross_correlation_debug.png")
            #     quit()

        cross = np.max(cross_correlation_array)
        i_max = shift_array[np.argmax(cross_correlation_array)]

        #print("cc:", cross)

        if plot:
            plt.figure(figsize=[20,3])
            plt.plot(self.t_array, cross_correlation_array)
            plt.xlabel("Time [ns]")
            plt.ylabel("Cross-correlation")
            plt.tight_layout()
            plt.savefig("cross_correlation.png")
            plt.close()

        
        return i_max, cross
    

    def chi2(self, data, signal):
        if isinstance(self.Vrms, np.ndarray) or isinstance(self.Vrms, list):
            sigma = self.Vrms[:,None]
        else:
            sigma = self.Vrms
        chi2 = np.sum((data - signal)**2 / sigma**2)
        #print("chi2",chi2)
        return chi2
    
    
    def get_efield(self, parameters, zenith_arrival, azimuth_arrival, use_channels, apply_filter=False):
        """
        Get the electric field in the two antennas for the given parameters.

        Parameters
        ----------
        parameters: np.ndarray
            Parameters of the signal model
            0: fluence theta #amplitude of the pulse
            1: fluence phi #polarization angle
            2: slope of the pulse
            3: phase of the pulse
            4: time of the pulse (maximum of absolute hilbert envelope)
            5: quadratic term

        Returns
        -------
        np.ndarray
            Electric field for given parameters
        """
        
        n_samples_time = self.n_samples
        sampling_rate = self.sampling_rate

        #amp_p0_theta = np.cos(parameters[1])**2 * parameters[0] #(parameters[0] * (parameters[1])#1e5
        #amp_p0_phi = np.sin(parameters[1])**2 * parameters[0] #parameters[0] * (1-parameters[1])
        amp_p0 = 1
        amp_p1 = parameters[2] #-2
        phase_p0 = parameters[3] #np.pi/2
        phase_p1 = - parameters[4] * 2 * np.pi # time offset in ns
        quadratic_term =  parameters[5] 
        quadratic_term_offset = self.filter_settings_high["passband"][0] # if self.filter_settings is not None else 0

        #efield_norm = get_analytic_pulse_freq(1, amp_p1, phase_p0, n_samples_time, sampling_rate, phase_p1=phase_p1, bandpass=None, quadratic_term=quadratic_term, quadratic_term_offset=quadratic_term_offset)
        #fluence_norm = np.sum((efield_norm * self.filter)**2)
        # amp_p0_theta = np.cos(parameters[1])**2 * parameters[0] / fluence_norm
        # amp_p0_phi = np.sin(parameters[1])**2 * parameters[0] / fluence_norm
        #A = np.sqrt(parameters[0] * fluence_norm)
        #amp_p0_phi = np.sin(parameters[1])**2 * parameters[0] / fluence_norm
        #amp_p0_theta = np.cos(parameters[1])**2 * A**2
        #amp_p0_phi = np.sin(parameters[1])**2 * A**2

        # Calculate the electric field:
        efield_norm = get_analytic_pulse_freq(amp_p0, amp_p1, phase_p0, n_samples_time, sampling_rate, phase_p1=phase_p1, bandpass=None, quadratic_term=quadratic_term, quadratic_term_offset=quadratic_term_offset)
        #efield_phi = get_analytic_pulse_freq(amp_p0_phi, amp_p1, phase_p0, n_samples_time, sampling_rate, phase_p1=phase_p1, bandpass=None, quadratic_term=quadratic_term, quadratic_term_offset=quadratic_term_offset)

        # Set the electric field:
        electric_field = ElectricField(use_channels, position=None, shower_id=None, ray_tracing_id=None)
        electric_field.set_frequency_spectrum(np.array([np.zeros_like(efield_norm), efield_norm, efield_norm]), self.sampling_rate)
        electric_field[efp.zenith] = zenith_arrival
        electric_field[efp.azimuth] = azimuth_arrival
        electric_field[efp.ray_path_type] = "direct"

        # Make a copy of the electric field and apply filter:
        efield_filtered = copy.copy(electric_field)
        sim_station = SimStation(0)
        sim_station.add_electric_field(efield_filtered)
        evt = NuRadioReco.framework.event.Event(1, 1)
        electricFieldBandPassFilter.run(evt, sim_station, det=None, **self.filter_settings_low)
        electricFieldBandPassFilter.run(evt, sim_station, det=None, **self.filter_settings_high)

        # Normalize the electric field to specified fluence (filtered):
        f_R, f_theta, f_phi = get_electric_field_energy_fluence(efield_filtered.get_trace(), efield_filtered.get_times())
        #f_R, f_theta_filt, f_phi_filt = get_electric_field_energy_fluence(efield_filtered.get_trace(), efield_filtered.get_times())
        if apply_filter:
            trace = efield_filtered.get_frequency_spectrum()
            # trace[1] *= np.sign(parameters[0]) * np.sqrt(np.abs(parameters[0]) / f_theta)
            # trace[2] *= np.sign(parameters[1]) * np.sqrt(np.abs(parameters[1]) / f_phi)
        else:
            trace = electric_field.get_frequency_spectrum()
        trace[1] *= np.sign(parameters[0]) * np.sqrt(np.abs(parameters[0]) / f_theta) #* np.sqrt(f_theta / f_theta_filt)
        trace[2] *= np.sign(parameters[1]) * np.sqrt(np.abs(parameters[1]) / f_phi)# * np.sqrt(f_phi / f_phi_filt)
        # print("fluence:", f_theta, f_phi, f_theta_filt, f_phi_filt)

        # quit()
        electric_field.set_frequency_spectrum(trace, self.sampling_rate)

        if 0:
            # print(efield_theta.shape)

            # print(efield_theta)

            # plt.figure()
            # plt.plot(range(len(efield_theta)), abs(efield_theta))
            # plt.show()

            # plt.figure()
            # plt.plot(self.t_array, np.fft.irfft(efield_theta))
            # plt.show()
            pass

        # print("max efield:", electric_field.get_trace_start_time(), electric_field.get_times()[np.argmax(np.sqrt(trace[1]**2+trace[1]**2))])
        # print(electric_field.get_times()[np.argmax(abs(trace[1]))], electric_field.get_times()[np.argmax(abs(trace[2]))])

        # plt.figure(figsize=[20,3])
        # plt.plot(electric_field.get_times(), electric_field.get_trace()[0], label="efield 1")
        # plt.plot(electric_field.get_times(), electric_field.get_trace()[1], label="efield 2")
        # plt.plot(electric_field.get_times(), electric_field.get_trace()[2], label="efield 3")
        # plt.savefig("efield.png")

        return electric_field
    

    def get_signal(self, parameters, det, station_id, use_channels, apply_filter="before"):
        """
        Get the signal in the two antennas for the given parameters.

        Parameters
        ----------
        parameters: np.ndarray
            Parameters of the signal model
            0: amplitude of the pulse
            1: polarization angle
            2: slope of the pulse
            3: phase of the pulse
            4: phase of the pulse (time / (2*np.pi))
            5: quadratic term
            6: zenith_arrival
            7: azimuth_arrival

        Returns
        -------
        np.ndarray
            Signal for given parameters
        """

        zenith_arrival = parameters[6]
        azimuth_arrival = parameters[7]

        electric_field = self.get_efield(parameters, zenith_arrival, azimuth_arrival, use_channels, apply_filter=(apply_filter=="before"))
        # plt.figure(figsize=[20,3])
        # plt.plot(electric_field.get_times(), electric_field.get_trace()[0], label="efield 1")   
        # plt.plot(electric_field.get_times(), electric_field.get_trace()[1], label="efield 2")
        # plt.plot(electric_field.get_times(), electric_field.get_trace()[2], label="efield 3")
        # plt.savefig("s2.png")
        #quit()
        # Add to sim station:
        sim_station = SimStation(station_id)
        sim_station.add_electric_field(electric_field)
        sim_station.set_is_cosmic_ray()
        sim_station[stnp.zenith] = zenith_arrival
        sim_station[stnp.azimuth] = azimuth_arrival
        #apply_det_response_sim(sim_station, self.detector, dict(), self.filter) # efield missing efp.ray_path_type, efp.zenith, and efp.azimuth

        evt = NuRadioReco.framework.event.Event(1, 1)
        station = NuRadioReco.framework.station.Station(station_id)
        station.add_sim_station(sim_station)
        station[stnp.zenith] = zenith_arrival
        station[stnp.azimuth] = azimuth_arrival
        efieldToVoltageConverter.run(evt, station, det, channel_ids=use_channels)

        # plt.figure(figsize=[20,3])
        # for i in range(3):
        #     plt.plot(np.arange(len(station.get_channel(i).get_trace()))/1, station.get_channel(i).get_trace(), label="channel {}".format(i))
        # plt.vlines(parameters[4] * 2 * np.pi, min(station.get_channel(0).get_trace()), max(station.get_channel(0).get_trace()), color="red", label="pulse start")
        # plt.savefig("debug1.png")

        # print("B")
        # print(station.get_channel(0).get_number_of_samples(), self.n_samples)

        if station.get_channel(use_channels[0]).get_number_of_samples() != self.n_samples:
            # offset = int(self.n_samples/2) #station.get_channel(self.use_channels[0]).get_number_of_samples() - self.n_samples - 750
            # channelLengthAdjuster.begin(number_of_samples=self.n_samples, offset=offset)
            # channelLengthAdjuster.run(evt, station, self.detector)

            for channel_id in use_channels:
                channel = station.get_channel(channel_id)
                channel.set_trace(channel.get_trace()[:self.n_samples], channel.get_sampling_rate())
                #channel.set_trace(channel.get_trace()[offset:offset+self.n_samples])
                #print(channel.get_trace().shape, self.n_samples, self.sampling_rate)


        if self.zenith_azimuth_free: # does this break signal window?

            for channel_id in use_channels:
                channel = station.get_channel(channel_id)

                empty_channel = NuRadioReco.framework.channel.Channel(channel_id)
                empty_channel.set_trace(np.zeros(self.n_samples), self.sampling_rate)
                empty_channel.set_trace_start_time(50*units.ns)
                empty_channel.add_to_trace(channel, raise_error=False)
                channel.set_trace(empty_channel.get_trace(), self.sampling_rate)
                channel.set_trace_start_time(50*units.ns)
                #print(channel.get_trace_start_time(), channel.get_trace().shape, self.n_samples, self.sampling_rate)

        # quit()

        # plt.figure(figsize=[20,3])
        # for i in range(3):
        #     plt.plot(station.get_channel(i).get_trace(), label="channel {}".format(i))
        # plt.savefig("debug2.png")
        
        # print("A")
        # print(electric_field.get_trace().shape, self.n_samples, self.sampling_rate)
        # print(sim_station.get_electric_fields()[0].get_trace().shape, self.n_samples, self.sampling_rate)
        # print(station.get_channel(0).get_trace().shape, self.n_samples, self.sampling_rate)

                
        # apply bandpass filter:
        #channelBandPassFilter.run(evt, station, self.detector, **self.filter_settings)
        if apply_filter == "after" or (self.filter_settings_high["filter_type"]=="rectangular" and self.filter_settings_low["filter_type"]=="rectangular"):
            channelBandPassFilter.run(evt, station, det, **self.filter_settings_low)
            channelBandPassFilter.run(evt, station, det, **self.filter_settings_high)

        
        traces = []
        for i in range(self.n_channels):
            traces.append(station.get_channel(use_channels[i]).get_trace())
        # trace_0 = station.get_channel(0).get_trace()
        # trace_1 = station.get_channel(1).get_trace()

        # print("max trace:", station.get_channel(0).get_trace_start_time(), station.get_channel(0).get_times()[np.argmax(abs(station.get_channel(0).get_trace()))])
        # print("max trace:", station.get_channel(1).get_trace_start_time(), station.get_channel(1).get_times()[np.argmax(abs(station.get_channel(1).get_trace()))])
        # plt.figure(figsize=[20,3])
        # for i in range(self.n_channels):
        #     plt.plot(station.get_channel(self.use_channels[i]).get_times(), traces[i], label="channel {}".format(i))
        # plt.savefig("s3.png")
        # quit()
        return np.array(traces)
    

    def calculate_polarization_and_fluence(self, parameters, det, use_channels, station_id):
        """
        Calculate the polarization of the signal from the given data.

        Parameters
        ----------
        data: np.ndarray
            Data from the two antennas

        Returns
        -------
        float
            Polarization of the signal
        """

        efield = self.get_efield(parameters[:6], parameters[6], parameters[7], use_channels, apply_filter=True)

        # Apply hard filter to avoid any unwanted signal at high frequencies:
        filter = np.ones_like(self.frequencies)
        filter[self.noise_model.spectra[0] < np.max(self.noise_model.spectra[0])/100] = 0
        max_freq = np.max(self.frequencies[filter == 1])
        sim_station = SimStation(station_id)
        sim_station.add_electric_field(efield)
        evt = NuRadioReco.framework.event.Event(1, 1)
        electricFieldBandPassFilter.run(evt, sim_station, det, passband=[0, max_freq], filter_type='rectangular')

        f_R, f_theta, f_phi = get_electric_field_energy_fluence(efield.get_trace(), efield.get_times())
        f_tot = f_R + f_theta + f_phi
        pol_angle = np.arctan2(f_phi**0.5, f_theta**0.5)

        return pol_angle, f_tot, f_theta, f_phi
    

    def reconstruct_signal(self, data, signal_function, parameters_initial, use_channels, second_order=True, signal_search_window=None):
        """
        Reconstruct the signal from the given data.

        Parameters
        ----------
        data: np.ndarray
            Data from the two antennas

        parameters_initial: np.ndarray
            Initial parameters for the reconstruction

        Returns
        -------
        np.ndarray
            Reconstructed signal
        """
        #parameters_initial = np.array([5, 10, -1, np.pi/2, 400, -10])
        signal_initial = signal_function(parameters_initial)

        
        # plt.figure(figsize=[10,3])
        # plt.plot(self.t_array, data[0,:])
        # plt.plot(self.t_array, signal_initial[0], label="signal 1")
        # plt.savefig("s1.png")
        # quit()
        dx_array = np.array([1e-3, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        fisher_information_matrix = self.noise_model.calculate_fisher_information_matrix(signal_function, parameters_initial, dx_array)
        # print()
        # plt.figure(figsize=[20,3])
        # plt.plot(self.t_array, signal_initial.T, label="signal 1")
        # plt.savefig("signal_initial.png")
        # quit()

        # plt.figure(figsize=[10,3])
        # plt.plot(self.frequencies, self.noise_psd.T, label="noise")
        # plt.savefig("noise_psd.png")
        # print(signal_initial)
        # print()
        # print(fisher_information_matrix)
        # print()
        f_i = np.linalg.pinv(fisher_information_matrix)
        errors = np.sqrt(np.diag(f_i))

        llh_initial = self.function_to_minimize_1(data, signal_initial)
        llh_initial_1 = self.noise_model.calculate_minus_two_delta_llh(data, signal_initial)
        if self.n_channels==2: i_max_inittial, cc_initial = self.cross_correlation(data, signal_initial, shift_array=self.i_shift_cc)
        llh_initial_2 = self.noise_model.calculate_minus_two_delta_llh(data, signal_initial)
        chi2_initial = self.chi2(data, signal_initial)

        print("Initial:")
        print(parameters_initial)
        #print(llh_initial, llh_initial_1, cc_initial, llh_initial_2, chi2_initial)

        bounds = np.array([
            (-1000, 1000),
            (-1000, 1000),
            (-100, -0.0001),
            (-3*np.pi, 3*np.pi),
            (-self.t_array[-1]*1.2, self.t_array[-1]*1.2),
            (-500, 0),
            (0, np.pi/2),
            (-2*np.pi, 2*np.pi)
            ])
        
        if signal_search_window is not None:
            # window = (self.t_array > signal_search_window[0]) & (self.t_array < signal_search_window[1])
            # self.t_array_window = self.t_array[window]
            # self.t_array_matched_filter = self.t_array_window - parameters_initial[4]
            # self.i_shift_cc = (self.t_array_matched_filter / self.sampling_rate).astype(int)
            parameters_initial[4] = (signal_search_window[0] + (signal_search_window[1] - signal_search_window[0]) / 2)
            self.t_array_matched_filter = np.arange(signal_search_window[0], signal_search_window[1], self.delta_t/4) - parameters_initial[4]
            self.i_shift_cc = (self.t_array_matched_filter / self.sampling_rate).astype(int)

        # print("ssw")
        # print(signal_search_window[0], signal_search_window[1])
        # #print(self.t_array_window[0], self.t_array_window[-1])
        # print(self.t_array_matched_filter[0], self.t_array_matched_filter[-1]) 
        # quit()

        reconstructor = TraceMinimizer(
            signal_function = signal_function,
            objective_function = self.function_to_minimize_1,
            parameters_initial = parameters_initial, # parameters_initial*1.1
            parameters_bounds = bounds,
        )
        if self.zenith_azimuth_free:
            reconstructor.fix_parameters([True, False, False, False, True, not(second_order), False, False])
        else:
            reconstructor.fix_parameters([True, False, False, False, True, not(second_order), True, True])
        reconstructor.set_scaling(errors)

        # if signal_search_window is not None:
        #     signal_window_mask = (self.t_array > signal_search_window[0]) & (self.t_array < signal_search_window[1])
        #     data_windowed = data * signal_window_mask
        #     m = reconstructor.reconstruct_event(data = data_windowed, method="minuit")
        # else:
        m = reconstructor.run_minimization(data=data, method="minuit")

        fitted_params = reconstructor.parameters
        llh_initial = self.function_to_minimize_1(data, signal_initial)
        llh_fit = reconstructor.result

        print(fitted_params, llh_initial, llh_fit)

        signal_fit = signal_function(fitted_params)

        # get time:
        if not self.use_chi2:
            t_offset, x = matched_filter(data, signal_fit, self.noise_psd, self.t_array_matched_filter, self.frequencies, self.n_channels)
        else:
            try:
                i_max, cross = self.cross_correlation(data, signal_fit, shift_array=self.i_shift_cc)
                t_offset = i_max / self.sampling_rate #self.t_array_matched_filter[i_max]
            except:
                print()
                print(self.t_array_matched_filter)
                print(self.i_shift_cc)
                print(self.t_array[self.i_shift_cc])
                print(i_max)
        parameters_adjusted = np.array([fitted_params[0], fitted_params[1], fitted_params[2], fitted_params[3], (fitted_params[4]+t_offset)%self.t_array[-1], parameters_initial[5], fitted_params[6], fitted_params[7]])
        signal_fit_adjusted = signal_function(parameters_adjusted) #adjust_time_offset_fft(signal_fit, t_offset, self.sampling_rate)
        signal_fit_adjusted = signal_fit_adjusted / np.max(signal_fit_adjusted) * np.max(data)
        

        # plot results:
        if 0:
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            ax[0].plot(self.t_array, data[0], label="data")
            ax[0].plot(self.t_array, signal_initial[0], label="initial")
            ax[0].plot(self.t_array, signal_fit[0], label="fit")
            ax[0].plot(self.t_array, signal_fit_adjusted[0], "--", label="fit adjusted")
            ax[0].set_ylabel("Voltage [V]")
            ax[0].legend()

            ax[1].plot(self.t_array, data[1], label="data")
            ax[1].plot(self.t_array, signal_initial[1], label="initial")
            ax[1].plot(self.t_array, signal_fit[1], label="fit")
            ax[1].plot(self.t_array, signal_fit_adjusted[1], "--", label="fit adjusted")
            ax[1].set_ylabel("Voltage [V]")
            ax[1].set_xlabel("Time [s]")
            ax[1].legend()
            plt.show()


        # fit time and amplitude:
        amplitude_correction = (np.max(data) / np.max(signal_fit))**2
        parameters_initial_2 = np.array([fitted_params[0]*amplitude_correction, fitted_params[1]*amplitude_correction, fitted_params[2], fitted_params[3], (parameters_initial[4]+t_offset)%self.t_array[-1], parameters_initial[5], parameters_initial[6], parameters_initial[7]])# if not self.zenith_azimuth_free else np.array([fitted_params[0]*amplitude_correction, fitted_params[1]*amplitude_correction, fitted_params[2], fitted_params[3], (parameters_initial[4]+t_offset)%self.t_array[-1], parameters_initial[5], fitted_params[-2], fitted_params[-1]])
        if self.zenith_azimuth_free:
            parameters_initial_2 = np.append(parameters_initial_2, [fitted_params[-2], fitted_params[-1]])
        print("parameters_initial_2", parameters_initial_2)
        print(bounds)
        print()
        #
        #polarization, fluence = self.calculate_polarization_and_fluence(parameters_initial_2)
        #
        if signal_search_window is not None:
            bounds[4] = (signal_search_window[0], signal_search_window[1])
            bounds[4][0] = bounds[4][0] - (bounds[4][1] - bounds[4][0]) / 2
            bounds[4][1] = bounds[4][1] + (bounds[4][1] - bounds[4][0]) / 2

        signal_initial_2 = signal_function(parameters_initial_2)

        reconstructor_2 = TraceMinimizer(
            signal_function = signal_function,
            objective_function = self.function_to_minimize_2,
            parameters_initial = parameters_initial_2,
            parameters_bounds = bounds,
        )

        if self.zenith_azimuth_free:
            reconstructor_2.fix_parameters([False, False, False, False, False, not(second_order), False, False])
        else:
            reconstructor_2.fix_parameters([False, False, False, False, False, not(second_order), True, True])

        fisher_information_matrix2 = self.noise_model.calculate_fisher_information_matrix(signal_function, fitted_params, dx_array)
        f_i2 = np.linalg.pinv(fisher_information_matrix2)
        errors2 = np.sqrt(np.diag(f_i2))
        reconstructor_2.set_scaling(errors2)

        m = reconstructor_2.run_minimization(data=data, method="minuit")

        fitted_params_2 = reconstructor_2.parameters
        llh_initial_2 = self.noise_model.calculate_minus_two_delta_llh(data, signal_initial_2)
        llh_fit_2 = reconstructor_2.result
        
        print("Results:")
        print(parameters_initial)
        print(fitted_params_2, llh_initial_1, llh_initial_2, llh_fit_2)

        signal_fit_2 = signal_function(fitted_params_2)

        #polarization, fluence, f_theta, f_phi = self.calculate_polarization_and_fluence(fitted_params_2, det,  use_channels, station_id)

        #print("Polarization and fluence:", polarization, fluence, f_theta, f_phi)
        print("fitted_parameters:", np.arctan2(np.abs(fitted_params_2[1])**0.5, np.abs(fitted_params_2[0])**0.5), np.abs(fitted_params_2[0])+np.abs(fitted_params_2[1]), np.abs(fitted_params_2[0]), np.abs(fitted_params_2[1]))
        
        polarization, fluence, f_theta, f_phi = np.arctan2(np.abs(fitted_params_2[1])**0.5, np.abs(fitted_params_2[0])**0.5), np.abs(fitted_params_2[0])+np.abs(fitted_params_2[1]), np.abs(fitted_params_2[0]), np.abs(fitted_params_2[1]) # make this not bound between 0 and 90 degrees

        fisher_information_matrix_fit = self.noise_model.calculate_fisher_information_matrix(signal_function, fitted_params_2, dx_array)
        f_i_fit = np.linalg.pinv(fisher_information_matrix_fit)
        errors_fit = np.sqrt(np.diag(f_i_fit))

        # error on polarization and fluence:
        # def polarization_wrapper(parameters):
        #     return self.calculate_polarization_and_fluence(parameters)[0]
        # def fluence_wrapper(parameters):
        #     return self.calculate_polarization_and_fluence(parameters)[1]
        # error_polarization = error_propagation(polarization_wrapper, fisher_information_matrix_fit, fitted_params_2, dx_array)
        # error_fluence = error_propagation(fluence_wrapper, fisher_information_matrix_fit, fitted_params_2, dx_array)
        f_theta_uncertainty = errors_fit[0]
        f_phi_uncertainty = errors_fit[1]
        error_fluence = np.sqrt(f_theta_uncertainty**2 + f_phi_uncertainty**2)
        error_polarization = np.sqrt( (np.sqrt(f_theta) / (2 * np.sqrt(f_phi) * (f_phi+f_theta)) )**2 * f_phi_uncertainty**2 + ( -np.sqrt(f_phi) / (2 * np.sqrt(f_theta) * (f_phi+f_theta)) )**2 * f_theta_uncertainty**2)

        # error_polarization = 0
        # error_fluence = 0
        # for i in range(len(errors_fit)):
        #     x_0_array = np.copy(fitted_params_2)
        #     x_1_array = np.copy(fitted_params_2)
        #     x_1_array[i] += dx_array[i]
        #     polarization_0, fluence_0 = self.calculate_polarization_and_fluence(x_0_array)
        #     polarization_1, fluence_1 = self.calculate_polarization_and_fluence(x_1_array)
        #     error_polarization += (polarization_1 - polarization_0)**2 / dx_array[i]**2 * errors_fit[i]**2
        #     error_fluence += (fluence_1 - fluence_0)**2 / dx_array[i]**2 * errors_fit[i]**2
        # error_polarization = np.sqrt(error_polarization)
        # error_fluence = np.sqrt(error_fluence)
        
        # plot results:
        if 1:
            fig, ax = plt.subplots(self.n_channels, 1, figsize=(10, self.n_channels*3))
            ax[0].plot(self.t_array, data[0]*2, label="data")
            ax[0].plot(self.t_array, signal_initial[0]*2, ls="--", label="initial")
            ax[0].plot(self.t_array, signal_fit[0], label="fit")
            ax[0].plot(self.t_array, signal_fit_adjusted[0], "--", label="fit adjusted")
            ax[0].plot(self.t_array, signal_initial_2[0], "y:", label="initial 2")
            ax[0].plot(self.t_array, signal_fit_2[0]*2, "k:", label="fit 2") #, parameters: {}".format(fitted_params_2))
            t_max = self.t_array[np.argmax(signal_fit_2[0])]
            ax[0].vlines([t_max+self.t_array_matched_filter[0],t_max+self.t_array_matched_filter[-1]], np.min(data[0]*2), np.max(data[0]*2), color="r", ls="--", label="matched filter")
            #ax[0].vlines([bounds[4][0], bounds[4][1]], np.min(data[0]*2), np.max(data[0]*2), color="m", ls="--", label="bounds")
            s0 = signal_function(np.array([fitted_params_2[0], fitted_params_2[1], fitted_params_2[2], fitted_params_2[3], bounds[4][0], fitted_params_2[5], fitted_params_2[6], fitted_params_2[7]]))
            t_max_bound_0 = self.t_array[np.argmax(s0[0])]
            s1 = signal_function(np.array([fitted_params_2[0], fitted_params_2[1], fitted_params_2[2], fitted_params_2[3], bounds[4][1], fitted_params_2[5], fitted_params_2[6], fitted_params_2[7]]))
            t_max_bound_1 = self.t_array[np.argmax(s1[0])]
            ax[0].vlines([t_max_bound_0, t_max_bound_1], np.min(data[0]*2), np.max(data[0]*2), color="b", ls="--", label="bounds")
            #ax[0].set_ylim([np.min(signal_fit_2), np.max(signal_fit_2)])
            ax[0].set_ylabel("Voltage [V]")
            if not self.use_chi2:
                ax[0].set_title(f"$-2\Delta$LLH: {llh_fit_2} \n parameters: {fitted_params_2}")
            else:
                ax[0].set_title(f"$\chi^2$: {llh_fit_2} \n parameters: {fitted_params_2}")
            ax[0].legend()

            ax[1].plot(self.t_array, data[1]*2, label="data")
            ax[1].plot(self.t_array, signal_initial[1]*2, ls="--", label="initial")
            ax[1].plot(self.t_array, signal_fit[1], label="fit")
            ax[1].plot(self.t_array, signal_fit_adjusted[1], "--", label="fit adjusted")
            ax[1].vlines(self.t_array[np.argmax(signal_fit_2[1])], np.min(data[0]*2), np.max(data[0]*2), color="k", ls="--", label="max fit 2")
            t_max = self.t_array[np.argmax(signal_fit_2[1])]
            ax[1].vlines([t_max+self.t_array_matched_filter[0],t_max+self.t_array_matched_filter[-1]], np.min(data[1]*2), np.max(data[1]*2), color="r", ls="--", label="matched filter")
            #ax[1].vlines([bounds[4][0], bounds[4][1]], np.min(data[1]*2), np.max(data[1]*2), color="m", ls="--", label="bounds")
            s0 = signal_function(np.array([fitted_params_2[0], fitted_params_2[1], fitted_params_2[2], fitted_params_2[3], bounds[4][0], fitted_params_2[5], fitted_params_2[6], fitted_params_2[7]]))
            t_max_bound_0 = self.t_array[np.argmax(s0[1])]
            s1 = signal_function(np.array([fitted_params_2[0], fitted_params_2[1], fitted_params_2[2], fitted_params_2[3], bounds[4][1], fitted_params_2[5], fitted_params_2[6], fitted_params_2[7]]))
            t_max_bound_1 = self.t_array[np.argmax(s1[1])]
            ax[1].vlines([t_max_bound_0, t_max_bound_1], np.min(data[1]*2), np.max(data[1]*2), color="b", ls="--", label="bounds")
            ax[1].plot(self.t_array, signal_initial_2[1], "y:", label="initial 2")
            ax[1].plot(self.t_array, signal_fit_2[1]*2, "k:", label="fit 2")

            if self.n_channels == 3:
                ax[2].plot(self.t_array, data[2]*2, label="data")
                ax[2].plot(self.t_array, signal_initial[2]*2, ls="--", label="initial")
                ax[2].plot(self.t_array, signal_fit[2], label="fit")
                ax[2].plot(self.t_array, signal_fit_adjusted[2], "--", label="fit adjusted")
                ax[2].vlines(self.t_array[np.argmax(signal_fit_2[1])], np.min(data[1]*2), np.max(data[1]*2), color="k", ls="--", label="max fit 2")
                ax[2].plot(self.t_array, signal_initial_2[2], "y:", label="initial 2")
                ax[2].plot(self.t_array, signal_fit_2[2]*2, "k:", label="fit 2")

            # p = fitted_params_2
            # p[4] = bounds[4][1] #self.t_array[np.argmax(data[1])]
            # signal = self.get_signal(p, apply_filter="after")
            # ax[1].plot(self.t_array, signal[1]*100, "k:", label="fit 123")

            #ax[1].set_ylim([np.min(signal_fit_2)*2, np.max(signal_fit_2)*2])
            ax[1].set_ylabel("Voltage [V]")
            ax[1].set_xlabel("Time [s]")
            ax[1].legend()
            plt.tight_layout()
            plt.savefig("debug33.png")
            #plt.savefig(f"plots/analytic_efield/efield/efield_core_{iE}_debug3.png")
            plt.show()
            plt.close()
        # print(bounds[4][1])
        # print(self.t_array[np.argmax(data[1])])
        # print(self.t_array[-1])
        # print(parameters_initial[4])
        # print(parameters_initial_2[4])
        # print(t_offset)
        # quit()
        # plot spectra:
        if 0:
            plt.figure()
            plt.plot(self.frequencies, self.noise_model.spectra[0], "k-", label="noise model")
            plt.plot(self.frequencies, self.noise_model.spectra[1], "k--")
            plt.plot(self.frequencies, np.abs(fft.time2freq(data[0], sampling_rate=self.sampling_rate)), "b-", label="data Theta")
            plt.plot(self.frequencies, np.abs(fft.time2freq(data[1], sampling_rate=self.sampling_rate)), "b--", label="data Phi")
            plt.plot(self.frequencies, np.abs(fft.time2freq(signal_initial[0], sampling_rate=self.sampling_rate)), "r-", label="initial Theta")
            plt.plot(self.frequencies, np.abs(fft.time2freq(signal_initial[1], sampling_rate=self.sampling_rate)), "r--", label="initial Phi")
            plt.plot(self.frequencies, np.abs(fft.time2freq(signal_fit_2[0], sampling_rate=self.sampling_rate)), "g-", label="fit Theta")
            plt.plot(self.frequencies, np.abs(fft.time2freq(signal_fit_2[1], sampling_rate=self.sampling_rate)), "g--", label="fit Phi")
            plt.hlines( np.max(self.noise_model.spectra[0])/100, 0, max(self.frequencies), "m", "--", label="threshold")

            plt.legend()
            plt.xlabel("Frequency [GHz]")
            plt.ylabel("Amplitude [V/GHz]")
            plt.yscale("log")
            plt.savefig("debug4.png")
            plt.show()
            plt.close()

            plt.figure()
            plt.plot(self.frequencies, self.noise_model.spectra[0], "k-", label="noise model")
            plt.plot(self.frequencies, self.noise_model.spectra[1], "k--")
            plt.plot(self.frequencies, np.abs(fft.time2freq(data[0], sampling_rate=self.sampling_rate)), "b-", label="data Theta")
            plt.plot(self.frequencies, np.abs(fft.time2freq(data[1], sampling_rate=self.sampling_rate)), "b--", label="data Phi")
            plt.plot(self.frequencies, np.abs(fft.time2freq(signal_initial[0], sampling_rate=self.sampling_rate)), "r-", label="initial Theta")
            plt.plot(self.frequencies, np.abs(fft.time2freq(signal_initial[1], sampling_rate=self.sampling_rate)), "r--", label="initial Phi")
            plt.hlines( np.max(self.noise_model.spectra[0])/100, 0, max(self.frequencies), "m", "--", label="threshold")
            plt.xlabel("Frequency [GHz]")
            plt.ylabel("Amplitude [V/GHz]")
            plt.axis([-10/len(self.frequencies), 500/len(self.frequencies), 0, 0.03])
            #plt.yscale("log")
            plt.show()
            plt.close()

        return llh_fit_2, polarization, fluence, fitted_params_2, errors_fit, error_polarization, error_fluence


    def get_random_efield(self, zenith_arrival, azimuth_arrival, use_channels):
        """
        Get a random electric field for testing.

        Returns
        -------
        np.ndarray
            Random electric field
        """

        bounds = np.array([
            (-10, 10),
            (-100, 100),
            (-10, -0.01),
            (0, 2*np.pi),
            (1000, 1000),
            (-10, 10)
            ])
            
        parameters = np.random.rand(6) * (bounds[:,1] - bounds[:,0]) + bounds[:,0]

        # We do not apply a filter here, since it is applied later in the code (true fluence will be wrong):
        efield = self.get_efield(parameters, zenith_arrival, azimuth_arrival, use_channels, apply_filter=False)

        return efield, parameters

    def get_random_event(self, station_id, use_channels):
        """
        Get a random event for testing.

        Returns
        -------
        np.ndarray
            Random event
        """

        zenith_arrival = np.random.random()*np.pi #self.zenith # these are not used
        azimuth_arrival = np.random.random()*2*np.pi #self.azimuth # these are not used

        electric_field, parameters = self.get_random_efield(zenith_arrival, azimuth_arrival, use_channels)

        # Add to sim station:
        sim_station = SimStation(station_id)
        sim_station.add_electric_field(electric_field)
        sim_station.set_is_cosmic_ray()
        sim_station[stnp.zenith] = zenith_arrival
        sim_station[stnp.azimuth] = azimuth_arrival
        #apply_det_response_sim(sim_station, self.detector, dict(), self.filter) # efield missing efp.ray_path_type, efp.zenith, and efp.azimuth

        evt = NuRadioReco.framework.event.Event(1, 1)
        station = NuRadioReco.framework.station.Station(station_id)
        station.add_sim_station(sim_station)
        station[stnp.zenith] = zenith_arrival
        station[stnp.azimuth] = azimuth_arrival
        efieldToVoltageConverter.run(evt, station, self.detector)

        # apply bandpass filter:
        #channelBandPassFilter.run(evt, station, self.detector, **self.filter_settings)

        trace_0 = station.get_channel(0).get_trace()
        trace_1 = station.get_channel(1).get_trace()

        parameters_true = np.array([parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], zenith_arrival, azimuth_arrival])

        return evt, station, sim_station, electric_field, parameters


def error_propagation(function, fisher_information_matrix, x_0, dx_):
    """
    Calculate the error on the function from the given fisher information matrix.

    Parameters
    ----------
    function: function
        Function to calculate the error on

    fisher_information_matrix: np.ndarray
        Fisher information matrix

    x_0: np.ndarray
        Parameters of the function

    dx: np.ndarray
        Step size for the derivative calculation

    Returns
    -------
    float
        Error on the function output
    """

    f_i = np.linalg.pinv(fisher_information_matrix)
    errors_x = np.sqrt(np.diag(f_i))

    dx = errors_x/1000

    # find derivatives:
    derivatives = np.zeros(len(x_0))
    for i in range(len(x_0)):
        x_0_array = np.copy(x_0)
        x_1_array = np.copy(x_0)
        x_1_array[i] += dx[i]
        derivatives[i] = (function(x_1_array) - function(x_0_array)) / dx[i]

    # calculate error:
    error = 0
    #error_matrix = np.zeros_like(f_i)
    for i in range(len(errors_x)):
        for j in range(len(errors_x)):
            error += derivatives[i] * f_i[i, j] * derivatives[j]
            #error_matrix[i, j] = derivatives[i] * f_i[i, j] * derivatives[j]
    error = np.sqrt(error)

    if 0:
        print("Errors:")
        print(errors_x)
        print("Derivatives:")
        print(derivatives)
        print("Error on function:", error)

        plt.figure()
        plt.imshow(np.log(np.abs(f_i)))
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.imshow(f_i / np.sqrt(np.diag(f_i)[:,np.newaxis]) / np.sqrt(np.diag(f_i)[np.newaxis,:]))
        plt.colorbar()
        plt.show()

        print("Error matrix:")
        #print(error_matrix)
        plt.figure()
        #plt.imshow(np.log(np.abs(error_matrix)))
        plt.colorbar()
        plt.show()

    return error