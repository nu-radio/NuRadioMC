"""
Reconstruction of cosmic-ray air shower signals

This module provides the `CREfieldReconstructor` class which can be used
to reconstruct the direction and polarization of a cosmic-ray air shower signal.
It currently works only if the arrival direction
of the signal can be assumed to be the same for all antennas
(e.g., shallow in-ice antennas that are all at the same refractive index).

This code is largely based on the `NuRadioReco.modules.voltageToAnalyticEfieldConverter`
module.

See Also
--------
NuRadioReco.modules.voltageToAnalyticEfieldConverter :
    older version of the cosmic-ray reconstruction code.

"""
import copy
from inspect import signature
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.constants
import scipy.signal

import radiotools.helper as hp
from radiotools import coordinatesystems
import NuRadioReco.framework.electric_field
from NuRadioReco.utilities import fft, trace_utilities, geometryUtilities, units, signal_processing, analytic_pulse
from NuRadioReco.detector import antennapattern
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.impulsiveSignalReconstructor import get_dt_correlation

logger = logging.getLogger('NuRadioReco.CREfieldReconstructor')
SPEED_OF_LIGHT = scipy.constants.c * units.m / units.s


class CREfieldReconstructor:
    """"
    Reconstruction class for cosmic-rays using shallow in-ice antennas

    """

    def __init__(self):
        self.__antenna_provider = antennapattern.AntennaPatternProvider()
        self.__amp_response = {}
        self.fixed_parameters = {param:False for param in signature(self.get_cosmic_ray_traces).parameters}

    def begin(self, evt, station, det, channels, extra_channels=[], bandpass=None, vrms=10*units.mV, debug=False, debug_folder='.'):
        self._debug = debug
        self._debug_folder = debug_folder
        self._station = station
        self._det = det
        self._vrms = vrms
        channel0 = station.get_channel(channels[0])
        self._freqs = channel0.get_frequencies()
        self._n_samples_time = channel0.get_number_of_samples()
        self._sampling_rate = channel0.get_sampling_rate()
        if bandpass is not None:
            if not isinstance(bandpass, dict):
                bandpass = dict(passband=bandpass)
            if 'filter_type' not in bandpass.keys():
                bandpass['filter_type'] = 'butterabs'
            if 'order' not in bandpass.keys():
                bandpass['order'] = 10

            self._filt = signal_processing.get_filter_response(self._freqs, **bandpass)
        else:
            self._filt = np.ones_like(self._freqs)

        self._filt[self._freqs > 1 * units.GHz] = 0 # the analytic parameterization is not valid for high frequencies, and the quadratic term may blow up

        # we fix the travel time delay at the channel with the largest amplitude to 0
        channels_sorted = np.array(channels)[np.argsort([np.max(np.abs(station.get_channel(channel).get_trace())) for channel in channels])[::-1]]
        # max_amp_channel = channels[np.argmax([np.max(np.abs(station.get_channel(channel).get_trace())) for channel in channels])]
        channels_sorted = np.concatenate([channels_sorted, extra_channels])
        self._channels_sorted = channels_sorted
        # store difference in trace start times
        trace_start_times = np.array([station.get_channel(channel).get_trace_start_time() for channel in channels_sorted])
        self._trace_start_times = trace_start_times - np.min(trace_start_times)

        self._channel_positions = np.array([det.get_relative_position(station.get_id(), channel) for channel in channels_sorted])
        self._channel_positions -= det.get_relative_position(station.get_id(), channels_sorted[0]).reshape((1,-1))

        # initialize antennas (initial load into memory takes long)
        for channel in channels_sorted:
            self.__amp_response[channel] = det.get_amplifier_response(self._station.get_id(), channel, self._freqs)
            self.__antenna_provider.load_antenna_pattern(det.get_antenna_model(station.get_id(), channel))

    def get_cosmic_ray_spectra(
            self, zenith, azimuth, amplitude, pol_angle, slope,
            phase_p0, phase_p1=0, quadratic_term=0, quadratic_term_offset=0.08, return_efield=False):
        """
        Return the cosmic-ray spectra

        Convenience function that returns the analytically-parameterized cosmic ray spectra
        for all channels as an array. For details of the parameterization, see [1]_.

        By default, this function returns the voltage spectra (including antenna and signal chain responses);
        to return the electric field spectra, set ``return_efield=True``.

        Parameters
        ----------
        zenith : float
            Zenith of the incoming cosmic ray
        azimuth : float
            Azimuth of the incoming cosmic ray
        amplitude : float
            Amplitude of the radio emission
        pol_angle : float
            Polarization angle, defined as arctan(E_phi / E_theta)
        slope : float
            Linear slope of the frequency spectrum
        phase_p0 : float
            Global phase of the spectra
        phase_p1 : float, default: 0
            Linear phase of the spectra (corresponds to an overall time shift of the pulse)
        quadratic_term : float, default: 0
            Quadratic term in the slope of the frequency spectra
        quadratic_term_offset : float, default: 80 * units.MHz
            Offset of the quadratic term (by default, 80 MHz)
        return_efield : bool, default: False
            If False (default), returns the voltage spectra, i.e.
            the electric field signals convolved with the antenna and detector signal chain
            responses.

            If True, returns the electric field spectra directly.

        Returns
        -------
        spectra : complex np.ndarray
            If ``return_efield==False``, an array of shape ``(n_channels, n_fft_samples)``
            containing the voltage spectra of each antenna, in descending order of SNR.

            If ``return_efield==True``, an array of shape ``(3, n_fft_samples)`` with the
            three polarization components ``(eR, eTheta, ePhi)`` of the electric field

        See Also
        --------
        get_cosmic_ray_traces : returns the cosmic ray signal in the time domain

        References
        ----------
        .. [1]


        """

        antenna_response = trace_utilities.get_efield_antenna_factor(
            self._station, self._freqs, self._channels_sorted, self._det, zenith, azimuth, self.__antenna_provider)
        time_delays = geometryUtilities.get_time_delay_from_direction(zenith, azimuth, self._channel_positions)
        efield = analytic_pulse.get_analytic_pulse_freq(
            100, -np.abs(slope), phase_p0=phase_p0, n_samples_time=self._n_samples_time, sampling_rate=self._sampling_rate,
            phase_p1=phase_p1, bandpass=None, quadratic_term=quadratic_term, quadratic_term_offset=quadratic_term_offset)

        A_theta = amplitude * np.cos(pol_angle)
        A_phi = amplitude * np.sin(pol_angle)

        if return_efield:
            return np.array([np.zeros_like(efield), A_theta*efield*self._filt, A_phi*efield*self._filt])

        # fold with the signal chain response
        res = np.zeros((len(self._channels_sorted), len(self._freqs)), dtype=complex)
        for iCh, channel in enumerate(self._channels_sorted):
            amp_response = self.__amp_response[channel]
            channel_spectrum = (
                (A_theta * efield*antenna_response[iCh][0] + A_phi*efield*antenna_response[iCh][1])
                * amp_response * self._filt * np.exp(-2j*np.pi*(time_delays[iCh] - self._trace_start_times[iCh])*self._freqs))
            res[iCh] = channel_spectrum
        return res

    def get_cosmic_ray_traces(
            self, zenith, azimuth, amplitude, pol_angle, slope,
            phase_p0, phase_p1=0, quadratic_term=0, quadratic_term_offset=0.08):
        spectra = self.get_cosmic_ray_spectra(
            zenith, azimuth, amplitude, pol_angle, slope,
            phase_p0, phase_p1, quadratic_term, quadratic_term_offset)
        traces = fft.freq2time(spectra, sampling_rate=self._sampling_rate)
        return traces

    def fix_parameters(self, fix_all=None, **parameters):
        """
        Convenience function to fix some of the fit parameters.

        Fixed fit parameters are not fitted.
        To see which fit parameters are used (and can be fixed),
        see `get_cosmic_ray_spectra`.

        Parameters
        ----------
        fix_all : bool, optional
            If ``True``, fix all fit parameters (so none is fitted);
            if ``False``, release all parameters to be fitted.
            Default ``None`` (i.e. only parameters included as keywords
            are affected by this method)
        **parameters
            Any (named) parameters accepted by the `get_cosmic_ray_spectra`
            method, with either the value ``True`` to fix the parameter
            (exclude from the fit) or ``False`` to include it in the fit.
            Overrides the setting from ``fix_all``, if present.

        Notes
        -----
        Scipy minimizers do not offer an interface to naturally 'fix'
        and 'release' parameters (e.g. to enable a step-wise) fit;
        this interface is inspired by the interface in iminuit,
        see https://scikit-hep.org/iminuit/notebooks/basic.html#Fixing-and-releasing-parameters

        """
        if fix_all is not None:
            for param in self.fixed_parameters.keys():
                self.fixed_parameters[param] = fix_all

        for param in parameters:
            if param not in self.fixed_parameters.keys():
                raise ValueError(f"No parameter with name {param} exists. Valid parameters are {self.fixed_parameters.keys()}")
            else:
                self.fixed_parameters[param] = parameters[param]

    def _minimize(self, x0, bounds=None, basinhopping=False, **scipy_kwargs):

        fit_param = np.array([not self.fixed_parameters[p] for p in signature(self.get_cosmic_ray_traces).parameters])
        fit_param = fit_param[:len(x0)]
        logger.debug('Starting fit with fixed parameters: {}'.format(self.fixed_parameters))

        if bounds is not None:
            bounds = scipy.optimize.Bounds(bounds[0][fit_param], bounds[1][fit_param])

        x0_tmp = np.copy(x0)

        chisq_const = np.sum(self._traces_data**2)

        def chisq(params, return_scale=False):
            x0_tmp[fit_param] = params
            traces_model = self.get_cosmic_ray_traces(*x0_tmp)
            crossterm = np.sum(self._traces_data * traces_model)
            modelterm = np.sum(traces_model**2)

            scale_min = crossterm / modelterm # overall scaling factor that minimizes chisq

            chisq = (scale_min**2 * modelterm - 2 * scale_min * crossterm + chisq_const) / self._vrms**2

            if return_scale:
                return chisq, scale_min

            return chisq

        if basinhopping:
            res = scipy.optimize.basinhopping(
                chisq, x0=np.array(x0)[fit_param],
                T=300, stepsize=0.1, niter_success=50, niter=250, interval=20, disp=self._debug,
                minimizer_kwargs=dict(bounds=bounds, **scipy_kwargs))
        else:
            res = scipy.optimize.minimize(chisq, x0=np.array(x0)[fit_param], bounds=bounds, **scipy_kwargs)

        _, rescaling_factor = chisq(res.x, return_scale=True) # update the overall amplitude scaling factor
        logger.debug(f"Rescaling amplitude by {rescaling_factor:.3g}")
        x0_tmp[fit_param] = res.x
        x0_tmp[2] *= rescaling_factor

        res.x = x0_tmp
        if not res.success:
            logger.warning(f"Fit failed: {res.message}")

        return res

    @register_run()
    def run(
            self, event, station, detector,
            use_MC_direction=False, include_quadratic_term=True,
            quadratic_term_offset = 80*units.MHz, basinhopping=False
        ):

        station = self._station
        channels = self._channels_sorted
        self._traces_data = np.array([
            fft.freq2time(station.get_channel(channel).get_frequency_spectrum() * self._filt, self._sampling_rate)
            for channel in channels])
        self._quadratic_term_offset = quadratic_term_offset
        minimizer_options = {} #dict(options=dict(maxiter=5000))#dict(method = 'Nelder-Mead')

        # obtain initial guess
        lags = scipy.signal.correlation_lags(self._n_samples_time, self._n_samples_time)
        lags = lags / self._sampling_rate
        p1_guess = np.pi*self._n_samples_time / self._sampling_rate # initially put reco traces in the middle
        if use_MC_direction:
            sim_station = station.get_sim_station()
            zenith = sim_station.get_parameter(stnp.zenith)
            azimuth = sim_station.get_parameter(stnp.azimuth)
        else:
            zenith_guess, azimuth_guess = 1, np.pi/2
            x0 = [zenith_guess, azimuth_guess, 1, np.pi/2, -2, 0, p1_guess, 0]
            zenith, azimuth = self._fit_direction_analytic(x0)

        try:
            sim_station = station.get_sim_station()
            zenith_sim = sim_station.get_parameter(stnp.zenith)
            azimuth_sim = sim_station.get_parameter(stnp.azimuth)
        except (KeyError, AttributeError):
            zenith_sim, azimuth_sim = np.nan, np.nan

        logger.debug(f"initial guess: {zenith/units.deg:.0f}, {azimuth/units.deg:.0f} (simulated: {zenith_sim/units.deg:.0f}, {azimuth_sim/units.deg:.0f})")
        traces_guess = self.get_cosmic_ray_traces(zenith, azimuth, 1, np.pi/2, -2, 0, phase_p1=p1_guess)

        iCh = 0 #channels.index(self._channels_sorted[0])
        amplitude_guess = np.max(self._traces_data[iCh]) / np.max(traces_guess[iCh])
        p1_guess += lags[np.argmax(hp.get_normalized_xcorr(traces_guess[iCh], self._traces_data[iCh]))] * 2*np.pi
        logger.debug(f"amplitude: {amplitude_guess:.3g}, p1_shift: {p1_guess:.1f}")
        logger.debug(f"{np.max(traces_guess[iCh]) , np.max(self._traces_data[iCh])}")
        traces_guess = self.get_cosmic_ray_traces(zenith, azimuth, amplitude_guess, np.pi/2, -2, 0, phase_p1=p1_guess)

        ## Iteration 1: fit slope and overall phase
        self.fix_parameters(zenith=True, azimuth=True, quadratic_term=True, amplitude=True, pol_angle=True)

        x0 = [zenith, azimuth, amplitude_guess, np.pi/2, -2, 0, p1_guess, 0]
        bounds = -np.inf*np.ones_like(x0), np.inf*np.ones_like(x0)
        bounds[1][4] = -1e-4
        res = self._minimize(x0=x0, **minimizer_options, bounds=bounds)
        logger.debug(f"First iteration: {res.fun:.3g}, {res.x}")

        ## Iteration 2: Fit slope, overall phase, zenith and azimuth
        ## Skipped if fixed to Monte-Carlo direction
        if not use_MC_direction:
            self.fix_parameters(zenith=False, azimuth=False)
            res = self._minimize(res.x)
            logger.debug(f"Second iteration: {res.fun:.3g}, {res.x}")

        ## Iteration 3: Include polarization angle and overall time shift
        self.fix_parameters(pol_angle=False, phase_p1=False)

        res = self._minimize(x0=res.x, **minimizer_options, bounds=bounds)
        logger.debug(f"Third iteration: {res.fun:.3g}, {res.x}")

        ## We may get stuck close to the initial value of A_theta=0
        ## If this is the case, we manually explore larger values of A_theta
        if np.abs(res.x[3] - np.pi/2) < 1e-2: # we may not have explored the polarization space properly:
            logger.debug('Additional iterations to explore polarization space more...')
            for pol_angle in [80*units.deg, 100*units.deg]:
                x0 = list(res.x)
                x0[3] = pol_angle
                res1 = self._minimize(x0=x0, **minimizer_options, bounds=bounds)
                if res1.fun < res.fun:
                    res = res1
                    logger.debug(f'Updated fit result: {res.fun:.3g}, {res.x}')

        if include_quadratic_term:
            ## We fit the quadratic term in two steps:
            ## First, we fit the quadratic term only, keeping everything else fixed
            self.fix_parameters(True)
            self.fix_parameters(quadratic_term=False)

            x0 = list(res.x)
            bounds[0][-1] = np.min([-2, x0[4]]) # we limit the quadratic term to be smaller than the linear one
            bounds[1][-1] = np.max([2, -x0[4]])
            res_quadratic = self._minimize(x0=res.x, **minimizer_options, bounds=bounds)

            ## and then we fit everything together
            self.fix_parameters(False)
            self.fix_parameters(amplitude=True) # amplitude is determined exactly, no need to fit
            res1 = self._minimize(x0=res_quadratic.x, **minimizer_options, bounds=bounds)

        if use_MC_direction:
            res = res1
        else:
            # we check the direction once more - it's possible we ended up in a local minimum
            x0 = copy.copy(res1.x)
            zenith, azimuth = self._fit_direction_analytic(x0)
            x0[:2] = zenith, azimuth
            res2 = self._minimize(x0=x0, **minimizer_options, bounds=bounds, method='Nelder-Mead')
            if res2.fun < res1.fun:
                logger.debug("Found new minimum for different direction")
                res = res2
                if not res.success:
                    res2 = self._minimize(x0=res.x, bounds=bounds, **minimizer_options)
                    if res2.fun < res.fun:
                        logger.debug('Additional BFGS minimization improved minimum.')
                        res = res2
            else:
                res = res1

        logger.debug(f"Final result: {res.fun:.3g}, {res.x}")

        if basinhopping:
            logger.info('Start additional basinhopping step to find global minimum...')
            res_bh = self._minimize(x0=res.x, bounds=bounds, basinhopping=True, **minimizer_options)
            logger.debug(f"After basinhopping: {res.fun:.3g}, {res.x}")
            if not res.success:
                logger.warning(f'!!! Final fit result may not be reliable: {res.message}')

            if res_bh.fun < res.fun:
                res = res_bh
            else:
                logger.warning('Basinhopping was a waste of time')


        ### store results - mostly taken from ``NuRadioReco.modules.voltageToAnalyticEfieldConverter``
        zenith, azimuth, amplitude, pol_angle, slope, phase_p0, phase_p1, quadratic_term = res.x

        station_trace = self.get_cosmic_ray_spectra(*res.x, return_efield=True)

        electric_field = NuRadioReco.framework.electric_field.ElectricField(channels)
        electric_field.set_frequency_spectrum(station_trace, self._sampling_rate)
        energy_fluence = trace_utilities.get_electric_field_energy_fluence(electric_field.get_trace(), electric_field.get_times())
        electric_field.set_parameter(efp.signal_energy_fluence, energy_fluence)
        # electric_field.set_parameter_error(efp.signal_energy_fluence, np.array([0, Atheta_error, Aphi_error]))
        electric_field.set_parameter(efp.cr_spectrum_slope, -np.abs(slope))
        electric_field.set_parameter(efp.zenith, zenith)
        electric_field.set_parameter(efp.azimuth, azimuth)
        station.set_parameter(stnp.zenith, zenith)
        station.set_parameter(stnp.azimuth, azimuth)

        # pol_angle = np.arctan2(A_phi, A_theta)
        # pol_angle_error = 1. / (x ** 2 + y ** 2) * (y ** 2 * sx ** 2 + x ** 2 + sy ** 2) ** 0.5  # gaussian error propagation
        logger.info("polarization angle = {:.1f}".format(pol_angle / units.deg))
        electric_field.set_parameter(efp.polarization_angle, pol_angle)
        # electric_field.set_parameter_error(efp.polarization_angle, pol_angle_error)

        site = self._det.get_site(station.get_id())
        exp_efield = hp.get_lorentzforce_vector(zenith, azimuth, hp.get_magnetic_field_vector(site))
        cs = coordinatesystems.cstrafo(zenith, azimuth, site=site)
        exp_efield_onsky = cs.transform_from_ground_to_onsky(exp_efield)
        exp_pol_angle = np.arctan2(exp_efield_onsky[2], exp_efield_onsky[1])
        logger.info("expected polarization angle = {:.1f}".format(exp_pol_angle / units.deg))
        electric_field.set_parameter(efp.polarization_angle_expectation, exp_pol_angle)
        electric_field.set_parameter(efp.cr_spectrum_quadratic_term, quadratic_term)

        station.add_electric_field(electric_field)

        if self._debug:
            traces_guess = self.get_cosmic_ray_traces(*res.x, quadratic_term_offset=self._quadratic_term_offset)
            n_rows = int(np.ceil(len(traces_guess)/3))
            fig, axs = plt.subplots(2*n_rows,3,figsize=(12,4*n_rows), sharey='row', height_ratios=n_rows*(2,1), layout='constrained')

            for i in range(len(traces_guess)):
                axs[2*(i//3), i%3].plot(np.arange(self._n_samples_time)/self._sampling_rate, self._traces_data[i], color='k', lw=.5)
                axs[2*(i//3), i%3].plot(np.arange(self._n_samples_time)/self._sampling_rate, traces_guess[i], lw=.5, alpha=.75, color='orange')
                axs[2*(i//3), i%3].set_title(f'Ch. {self._channels_sorted[i]}: $\chi^2 = {np.sum( (self._traces_data[i]-traces_guess[i])**2 / self._vrms**2):.0f} / {len(traces_guess[i])}$')
                axs[2*(i//3), i%3].set_xlabel('Time [ns]')
                axs[2*(i//3), 0].set_ylabel('Voltage [V]')

                corr = hp.get_normalized_xcorr(self._traces_data[i], traces_guess[i])
                axs[2*(i//3) + 1, i%3].plot(lags, corr, lw=.5)
                axs[2*(i//3) + 1, i%3].axvline(lags[np.argmax(corr)], ls=':', color='r')
                axs[2*(i//3) + 1, i%3].set_xlim(-100, 100)
                axs[2*(i//3) + 1, i%3].set_xlabel('Shift [ns]')
                axs[2*(i//3) + 1, 0].set_ylabel('Correlation')

            fig.suptitle(
                f'S{station.get_id()}R{event.get_run_number()}E{event.get_id()}: $\\theta = {zenith/units.deg:.1f}^\circ, $'
                + f'$\phi = {azimuth/units.deg:.1f}^\circ, $'
                + f'$\\theta_\mathrm{{pol}} = {pol_angle/units.deg:.1f}^\circ$ '
                + f' (exp: ${exp_pol_angle/units.deg:.1f}^\circ$)'
                )
            plt.savefig(os.path.join(self._debug_folder, f'S{station.get_id()}R{event.get_run_number()}E{event.get_id()}.pdf'))
            plt.close()

        return res

    def _fit_direction_analytic(self, params, tol=2*units.deg, maxiter=5):
        """
        Obtain a first guess for the zenith and azimuth

        Uses the maximum correlation and an analytic plane wave
        solution to estimate the source zenith and azimuth. As the
        analytical signal depends on the input zenith and azimuth,
        this can be iterated for ``maxiter`` iterations until a tolerance
        ``tol`` is achieved.

        Returns
        -------
        zenith, azimuth: tuple of floats
        """

        n_iterations = 0
        zenith_guess, azimuth_guess = params[:2] # zenith and azimuth are the first two entries
        channels = [self._station.get_channel(channel_id) for channel_id in self._channels_sorted[:3]]

        while n_iterations < maxiter:
            params[0:2] = zenith_guess, azimuth_guess
            traces_guess = self.get_cosmic_ray_traces(*params)[:3]
            delta_t_geometry = geometryUtilities.get_time_delay_from_direction(zenith_guess, azimuth_guess, self._channel_positions)
            templates = [np.roll(t, -int(np.round(delta_t * self._sampling_rate))) for t, delta_t in zip(traces_guess, delta_t_geometry)]

            t_shifts_guess, corr = get_dt_correlation(channels, self._channel_positions, templates=templates, full_output=True)
            max_corr = np.sum(corr)
            zenith = np.nan
            i = 0

            # the best guess (by correlation) may not have a solution; in that case, we check the
            # 'second best' guess by taking the next-highest correlation for the lowest SNR channel
            while np.isnan(zenith):
                zenith, azimuth = geometryUtilities.analytic_plane_wave_fit(t_shifts_guess, self._channel_positions)
                i += 1

            if np.isnan(zenith): # no solution was found in the end, so we just return the initial guess
                zenith = zenith_guess
                azimuth = azimuth_guess
                break
            elif hp.get_angle(hp.spherical_to_cartesian(zenith, azimuth), hp.spherical_to_cartesian(zenith_guess, azimuth_guess)) < tol:
                logger.debug(f"Converged after {n_iterations} iterations - ({zenith/units.deg:.0f}, {azimuth/units.deg:.0f})")
                break
            else:
                logger.debug(f"Iteration {n_iterations} - ({zenith/units.deg:.0f}, {azimuth/units.deg:.0f}) (total correlation {np.sum(max_corr):.2f})")
                zenith_guess = zenith
                azimuth_guess = azimuth

            n_iterations += 1

        return zenith, azimuth
