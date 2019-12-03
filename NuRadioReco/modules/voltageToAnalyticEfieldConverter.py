from __future__ import absolute_import, division, print_function, unicode_literals
from NuRadioReco.modules.base.module import register_run
import os
import time
import numpy as np
from numpy.polynomial import polynomial as poly
from scipy import signal
from scipy.signal import correlate
from scipy import optimize as opt
import matplotlib.pyplot as plt

from radiotools import plthelpers as php
from radiotools import helper as hp
from radiotools import coordinatesystems

from NuRadioReco.detector import detector
from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units, fft, trace_utilities
from NuRadioReco.utilities import analytic_pulse as pulse
from NuRadioReco.modules.voltageToEfieldConverter import get_array_of_channels

from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import NuRadioReco.framework.electric_field

import logging
logger = logging.getLogger('voltageToAnalyticEfieldConverter')


def covariance(function, vmin, up, fast=False):
    """
    Numerically compute the covariance matrix from a chi^2 or -logLikelihood function.

    Parameters
    ----------
    function: function-like
      The function may accept only a vector argument and has to return a scalar.
    vmin: array of floats
      Position of the minimum.
    up: float
      Threshold value to pass when climbing uphill.
      up = 1   for a chi^2 function
      up = 0.5 for a -logLikelihood function
    fast: boolean
      If true invert hesse matrix at the minimum, use this if computing function is expensive.

    Examples
    --------
    >>> cov = ((2.0,0.2),(0.2,2.0))
    >>> invcov = np.linalg.inv(cov)
    >>> xs = np.array((1.0,-1.0))
    >>> def ChiSquare(pars, grad = None): return np.dot(xs-pars,np.dot(invcov,xs-pars))
    >>> def NegLogLike(pars, grad = None): return 0.5*ChiSquare(pars)
    >>> covariance(ChiSquare, xs, 1.0)
    array([[ 2. ,  0.2],
           [ 0.2,  2. ]])
    >>> covariance(ChiSquare, xs, 1.0, fast=True)
    array([[ 2. ,  0.2],
           [ 0.2,  2. ]])
    >>> covariance(NegLogLike, xs, 0.5)
    array([[ 2. ,  0.2],
           [ 0.2,  2. ]])
    >>> covariance(NegLogLike, xs, 0.5, fast=True)
    array([[ 2. ,  0.2],
           [ 0.2,  2. ]])

    Notes
    -----
    The algorithm is slow (it needs many function evaluations), but robust.
    The covariance matrix is derived by explicitly following the chi^2
    or -logLikelihood function uphill until it crosses the 1-sigma contour.

    The fast alternative is to invert the hessian matrix at the minimum.

    Author
    ------
    Hans Dembinski <hans.dembinski@kit.edu>
    """

    from scipy.optimize import brentq

    class Func:

        def __init__(self, function, vmin, up):
            self.dir = np.zeros_like(vmin)
            self.up = up
            self.vmin = vmin
            self.fmin = function(vmin)
            self.func = function

        def __call__(self, x):
            return self.func(self.vmin + x * self.dir) - self.fmin - self.up

        def SetDirection(self, i, j):
            self.dir *= 0
            self.dir[abs(i)] = 1 if i >= 0 else -1
            self.dir[abs(j)] = 1 if j >= 0 else -1

        def GetBoundary(self, sign):
            eps = np.sqrt(np.finfo(np.double).eps)
            h = eps
            x0 = abs(np.dot(self.vmin, self.dir))

            def IsNonsense(x):
                return np.isnan(x) or np.isinf(x)

            def x(h):
                return sign * (h * x0 if x0 != 0 else h)

            while True:
                # (1) do smallest possible step first,
                #     then grow exponentially until zero+ is crossed,

                if IsNonsense(x(h)):
                    raise StandardError("profile does not cross fmin + up")

                t = self(x(h))

                if IsNonsense(t):
                    # (2) if stepped into nonsense region (infinite, nan, ...),
                    #     do bisection search towards last valid step
                    a = h / 8.0
                    b = h
                    while True:
                        if 2 * (b - a) < eps * (b + a):
                            raise StandardError("profile does not cross fmin + up")
                    h = (a + b) / 2.0
                    t = self(x(h))

                    if IsNonsense(t):
                        b = h
                        continue

                    if t < 0:
                        a = h
                        continue

                    return x(h)

                if t > 0:
                    return x(h)

                h *= 16

    n = len(vmin)

    if fast:
        from pyik.numpyext import hessian
        releps = 1e-3
        dvmin = vmin * releps
        dvmin[dvmin == 0] = releps
        a = hessian(function, vmin, dvmin) / up
    else:
        # Ansatz: (f(r) - fmin)/up = 1/2 r^T C r == 1
        # Diagonal elements:
        # 1 != 1/2 sum_{ij} delta_ik x delta_jk x C_ij
        #    = x^2/2 C_kk
        # => C_kk = 2/x^2
        # Off-diagonal elements:
        # 1 != 1/2 x (delta_ik + delta_il) C_ij x (delta_jk + delta_jl)
        #    = x^2/2 (C_kk + C_kl + C_lk + C_ll) = x^2/2 (2 C_kl + C_kk + C_ll)
        # => C_kl = 0.5 * (2/x^2 - C_kk - C_ll)

        func = Func(function, vmin, up)
        d = np.empty((n, n))
        for i in xrange(n):
            func.SetDirection(i, i)

            xu = func.GetBoundary(+1)
            t = func(-xu)
            xd = -xu if not np.isinf(t) and t > 0.0 else func.GetBoundary(-1)

            x1 = +brentq(func, 0, xu)
            x2 = -brentq(func, xd, 0)
            x = 0.5 * (x1 + x2)

            if x < 0:
                raise StandardError("x may not be negative")

            d[i, i] = x

        for i in xrange(n - 1):
            for j in xrange(i + 1, n):
                func.SetDirection(i, j)

                xu = func.GetBoundary(+1)
                t = func(-xu)
                xd = -xu if not np.isinf(t) and t > 0.0 else func.GetBoundary(-1)

                x1 = +brentq(func, 0, xu)
                x2 = -brentq(func, xd, 0)
                x = 0.5 * (x1 + x2)

                if x < 0:
                    raise StandardError("x may not be negative")

                # check whether x is in possible range
                a = d[i, i]
                b = d[j, j]
                xmax = np.inf if a <= b else 1.0 / (1.0 / b - 1.0 / a)
                xmin = 1.0 / (1.0 / b + 1.0 / a)

                if x <= xmin:
                    logger.warning("covariance(...):", xmin, "<", x, "<", xmax, "violated")
                    x = xmin * 1.01
                if x >= xmax:
                    logger.warning("covariance(...):", xmin, "<", x, "<", xmax, "violated")
                    x = xmax * 0.99

                d[i, j] = d[j, i] = x

        a = 2.0 / d ** 2

        for i in xrange(n - 1):
            for j in xrange(i + 1, n):
                a[i, j] = a[j, i] = 0.5 * (a[i, j] - a[i, i] - a[j, j])

    # Beware: in case of a chi^2 we calculated
    # t^2 = (d^2 chi^2 / d par^2)^{-1},
    # while s^2 = (1/2 d^2 chi^2 / d par^2)^{-1} is correct,
    # thus s^2 = 2 t^2
    cov = 2.0 * np.linalg.inv(a)

    # first aid, if 1-sigma contour does not look like hyper-ellipsoid
    for i in xrange(n):
        if cov[i, i] < 0:
            logger.warning("covariance(...): error, cov[%i,%i] < 0, returning zero" % (i, i))
            for j in xrange(n):
                cov[i, j] = 0

    return cov


def stacked_lstsq(L, b, rcond=1e-10):
    """
    Solve L x = b, via SVD least squares cutting of small singular values
    L is an array of shape (..., M, N) and b of shape (..., M).
    Returns x of shape (..., N)
    """
    u, s, v = np.linalg.svd(L, full_matrices=False)
    s_max = s.max(axis=-1, keepdims=True)
    s_min = rcond * s_max
    inv_s = np.zeros_like(s)
    inv_s[s >= s_min] = 1 / s[s >= s_min]
    x = np.einsum('...ji,...j->...i', v,
                  inv_s * np.einsum('...ji,...j->...i', u, b.conj()))
    return np.conj(x, x)


class voltageToAnalyticEfieldConverter:
    """
    reconstucts the electric-field by foward folding an analytic pulse function through the antenna

    This module works only for cosmic rays so far. The cosmic-ray radio pulse
    can be described in Fourier space with a simple exponential function for
    the magnitude as a function of frequency.
    The phase is set to zero. A slope in the phase spectrum corresponds to a
    translation in time. In each iteration step, the time shift is set to the
    shift that results in the maximal cross correlation.

    The module also calculates the polarization angle and the energy fluence
    from the fit parameters.
    """

    def __init__(self):
        self.__counter = 0
        self.begin()

    def begin(self):
        """
        begin method. This function is executed before the event loop.

        The antenna pattern provider is initialized here.
        """
        self.antenna_provider = antennapattern.AntennaPatternProvider()
        pass

    @register_run()
    def run(self, evt, station, det, debug=False, debug_plotpath=None,
            use_channels=[0, 1, 2, 3],
            bandpass=[100 * units.MHz, 500 * units.MHz],
            use_MC_direction=False):
        """
        run method. This function is executed for each event

        Parameters
        ---------
        evt
        station
        det
        debug: bool
            if True debug plotting is enables
        debug_plotpath: string or None
            if not None plots will be saved to a file rather then shown. Plots will
            be save into the `debug_plotpath` directory
        use_channels: array of ints
            the channel ids to use for the electric field reconstruction
            default: 0 - 3
        bandpass: [float, float]
            the lower and upper frequecy for which the analytic pulse is calculated.
            A butterworth filter of 10th order and a rectangular filter is applied.
            default 100 - 500 MHz
        use_MC_direction: bool
            use simulated direction instead of reconstructed direction
        """
        self.__counter += 1
        event_time = station.get_station_time()
        station_id = station.get_id()
        logger.info("event {}, station {}".format(evt.get_id(), station_id))
        if use_MC_direction and (station.get_sim_station() is not None):
            zenith = station.get_sim_station()[stnp.zenith]
            azimuth = station.get_sim_station()[stnp.azimuth]
            sim_present = True
        else:
            logger.warning("Using reconstructed angles as no simulation present")
            zenith = station[stnp.zenith]
            azimuth = station[stnp.azimuth]
            sim_present = False

        efield_antenna_factor, V, V_timedomain = get_array_of_channels(station, use_channels,
                                                                       det, zenith, azimuth, self.antenna_provider,
                                                                       time_domain=True)
        sampling_rate = station.get_channel(0).get_sampling_rate()
        n_samples_time = V_timedomain.shape[1]

        debug_obj = 0
        noise_RMS = det.get_noise_RMS(station.get_id(), 0)

        def obj_xcorr(params):
            if(len(params) == 3):
                slope, ratio2, phase2 = params
                ratio = (np.arctan(ratio2) + np.pi * 0.5) / np.pi  # project -inf..inf on 0..1
            elif(len(params) == 2):
                slope, ratio2 = params
                phase2 = 0
                ratio = (np.arctan(ratio2) + np.pi * 0.5) / np.pi  # project -inf..inf on 0..1
            elif(len(params) == 1):
                phase2 = 0
                ratio = 0
                slope = params[0]
            phase = np.arctan(phase2)  # project -inf..+inf to -0.5 pi..0.5 pi

            analytic_pulse_theta = pulse.get_analytic_pulse_freq(ratio, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
            analytic_pulse_phi = pulse.get_analytic_pulse_freq(1 - ratio, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
            chi2 = 0

            n_channels = len(V_timedomain)
            analytic_traces = np.zeros((n_channels, n_samples_time))
            positions = np.zeros(n_channels, dtype=np.int)
            max_xcorrs = np.zeros(n_channels)
            # first determine the position with the larges xcorr
            for iCh, trace in enumerate(V_timedomain):
                analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([analytic_pulse_theta, analytic_pulse_phi]), axis=0)
                analytic_traces[iCh] = fft.freq2time(analytic_trace_fft, sampling_rate)
                xcorr = np.abs(hp.get_normalized_xcorr(trace, analytic_traces[iCh]))
                positions[iCh] = np.argmax(np.abs(xcorr)) + 1
                max_xcorrs[iCh] = xcorr.max()
                chi2 -= xcorr.max()
            logger.debug("ratio = {:.2f}, slope = {:.4g}, phase = {:.0f} ({:.4f}), chi2 = {:.4g}".format(ratio, slope, phase / units.deg, phase2, chi2))
            return chi2

        def obj_amplitude(params, slope, phase, pos, debug_obj=0):
            if(len(params) == 2):
                ampPhi, ampTheta = params
            elif(len(params) == 1):
                ampPhi = params[0]
                ampTheta = 0
            analytic_pulse_theta = pulse.get_analytic_pulse_freq(ampTheta, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
            analytic_pulse_phi = pulse.get_analytic_pulse_freq(ampPhi, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
            chi2 = 0

            if(debug_obj):
                fig, ax = plt.subplots(4, 2, sharex=True)

            n_channels = len(V_timedomain)
            analytic_traces = np.zeros((n_channels, n_samples_time))
            # first determine the position with the larges xcorr
            for iCh, trace in enumerate(V_timedomain):
                analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([analytic_pulse_theta, analytic_pulse_phi]), axis=0)
                analytic_traces[iCh] = fft.freq2time(analytic_trace_fft, sampling_rate)

                argmax = np.argmax(np.abs(trace))
                imin = np.int(argmax - 30 * sampling_rate)
                imax = np.int(argmax + 50 * sampling_rate)

                tmp = np.sum(np.abs(trace[imin:imax] - np.roll(analytic_traces[iCh], pos)[imin:imax]) / noise_RMS)
                chi2 += tmp ** 2
                if(debug_obj):
                    ax[iCh][0].plot(trace, label='measurement')
                    ax[iCh][0].plot(np.roll(analytic_traces[iCh], pos), '--', label='fit')
                    ax[iCh][1].plot(trace - np.roll(analytic_traces[iCh], pos), label='delta')
                    ax[iCh][1].set_xlim(imin, imax)
            logger.debug("amp phi = {:.4g}, amp theta = {:.4g} , chi2 = {:.2g}".format(ampPhi, ampTheta, chi2))
            if(debug_obj):
                fig.suptitle("amp phi = {:.4g}, amp theta = {:.4g} , chi2 = {:.2g}".format(ampPhi, ampTheta, chi2))
                fig.tight_layout()
                plt.show()
            return chi2

        def obj_amplitude_slope(params, phase, pos, compare='hilbert', debug_obj=0):
            ampPhi, ampTheta, slope = params
            analytic_pulse_theta = pulse.get_analytic_pulse_freq(ampTheta, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
            analytic_pulse_phi = pulse.get_analytic_pulse_freq(ampPhi, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
            chi2 = 0
            if(debug_obj and self.i_slope_fit_iterations % 25 == 0):
                fig, ax = plt.subplots(4, 2, sharex=False, figsize=(20, 10))

            n_channels = len(V_timedomain)
            analytic_traces = np.zeros((n_channels, n_samples_time))
            # first determine the position with the larges xcorr
            channel_max = 0
            for trace in V_timedomain:
                if np.max(np.abs(trace)) > channel_max:
                    channel_max = np.max(np.abs(trace))
                    argmax = np.argmax(np.abs(trace))
                    imin = np.int(max(argmax - 50 * sampling_rate, 0))
                    imax = np.int(argmax + 50 * sampling_rate)
            for iCh, trace in enumerate(V_timedomain):
                analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([analytic_pulse_theta, analytic_pulse_phi]), axis=0)
                analytic_traces[iCh] = fft.freq2time(analytic_trace_fft, sampling_rate)
                if compare == 'trace':
                    tmp = np.sum(np.abs(trace[imin:imax] - np.roll(analytic_traces[iCh], pos)[imin:imax]) ** 2) / noise_RMS ** 2
                elif compare == 'abs':
                    tmp = np.sum(np.abs(np.abs(trace[imin:imax]) - np.abs(np.roll(analytic_traces[iCh], pos)[imin:imax])) ** 2) / noise_RMS ** 2
                elif compare == 'hilbert':
                    tmp = np.sum(np.abs(np.abs(signal.hilbert(trace[imin:imax])) - np.abs(signal.hilbert(np.roll(analytic_traces[iCh], pos)[imin:imax]))) ** 2) / noise_RMS ** 2
                else:
                    raise NameError('Unsupported value for parameter "compare": {}. Value must be "trace", "abs" or "hilbert".'.format(compare))
                chi2 += tmp
                if(debug_obj and self.i_slope_fit_iterations % 25 == 0):
                    ax[iCh][1].plot(np.array(trace) / units.mV, label='measurement', color='blue')
                    ax[iCh][1].plot(np.roll(analytic_traces[iCh], pos) / units.mV, '--', label='fit', color='orange')
                    ax[iCh][1].plot(np.abs(signal.hilbert(trace)) / units.mV, linestyle=':', color='blue')
                    ax[iCh][1].plot(np.abs(signal.hilbert(np.roll(analytic_traces[iCh], pos))) / units.mV, ':', label='fit', color='orange')
                    # ax[iCh][1].plot(trace - np.roll(analytic_traces[iCh], pos), label='delta')
                    ax[iCh][0].plot(np.abs(fft.time2freq(trace, sampling_rate)) / units.mV, label='measurement')
                    ax[iCh][0].plot(np.abs(fft.time2freq(np.roll(analytic_traces[iCh], pos), sampling_rate)) / units.mV, '--', label='fit')
                    ax[iCh][0].set_xlim([0, 600])
                    ax[iCh][1].set_xlim([imin - 500, imax + 500])
                    ax[iCh][1].axvline(imin, linestyle='--', alpha=.8)
                    ax[iCh][1].axvline(imax, linestyle='--', alpha=.8)
            logger.debug("amp phi = {:.4g}, amp theta = {:.4g}, slope = {:.4g} chi2 = {:.8g}".format(ampPhi, ampTheta, slope, chi2))
            if(debug_obj and self.i_slope_fit_iterations % 25 == 0):
#                 ax[0][0].set_title("ratio = {:.2f} ({:.2f}), slope = {:.2g}, phase = {:.0f} ({:.2f}), chi2 = {:.2g}".format(ratio, ratio2, slope, phase / units.deg, phase2, chi2))
                fig.tight_layout()
                plt.show()
                self.i_slope_fit_iterations = 0
            self.i_slope_fit_iterations += 1
            return chi2

        def obj_amplitude_second_order(params, slope, phase, pos, compare='hilbert', debug_obj=0):
            ampPhi, ampTheta, second_order = params
            analytic_pulse_theta = pulse.get_analytic_pulse_freq(ampTheta, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass, quadratic_term=second_order, quadratic_term_offset=bandpass[0])
            analytic_pulse_phi = pulse.get_analytic_pulse_freq(ampPhi, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass, quadratic_term=second_order, quadratic_term_offset=bandpass[0])
            chi2 = 0
            if(debug_obj and self.i_slope_fit_iterations % 50 == 0):
                fig, ax = plt.subplots(5, 2, sharex=False, figsize=(20, 10))

            n_channels = len(V_timedomain)
            analytic_traces = np.zeros((n_channels, n_samples_time))
            # first determine the position with the larges xcorr
            channel_max = 0
            for trace in V_timedomain:
                if np.max(np.abs(trace)) > channel_max:
                    channel_max = np.max(np.abs(trace))
                    argmax = np.argmax(np.abs(trace))
                    imin = np.int(max(argmax - 50 * sampling_rate, 0))
                    imax = np.int(argmax + 50 * sampling_rate)
            for iCh, trace in enumerate(V_timedomain):
                analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([analytic_pulse_theta, analytic_pulse_phi]), axis=0)
                analytic_traces[iCh] = fft.freq2time(analytic_trace_fft, sampling_rate)
                if compare == 'trace':
                    tmp = np.sum(np.abs(trace[imin:imax] - np.roll(analytic_traces[iCh], pos)[imin:imax]) ** 2) / noise_RMS ** 2
                elif compare == 'abs':
                    tmp = np.sum(np.abs(np.abs(trace[imin:imax]) - np.abs(np.roll(analytic_traces[iCh], pos)[imin:imax])) ** 2) / noise_RMS ** 2
                elif compare == 'hilbert':
                    tmp = np.sum(np.abs(np.abs(signal.hilbert(trace[imin:imax])) - np.abs(signal.hilbert(np.roll(analytic_traces[iCh], pos)[imin:imax]))) ** 2) / noise_RMS ** 2
                else:
                    raise NameError('Unsupported value for parameter "compare": {}. Value must be "trace", "abs" or "hilbert".'.format(compare))
                chi2 += tmp
                if(debug_obj and self.i_slope_fit_iterations % 50 == 0):
                    ax[iCh][1].plot(np.array(trace) / units.mV, label='measurement', color='blue')
                    ax[iCh][1].plot(np.roll(analytic_traces[iCh], pos) / units.mV, '--', label='fit', color='orange')
                    ax[iCh][1].plot(np.abs(signal.hilbert(trace)) / units.mV, linestyle=':', color='blue')
                    ax[iCh][1].plot(np.abs(signal.hilbert(np.roll(analytic_traces[iCh], pos))) / units.mV, ':', label='fit', color='orange')
                    # ax[iCh][1].plot(trace - np.roll(analytic_traces[iCh], pos), label='delta')
                    ax[iCh][0].plot(np.abs(fft.time2freq(trace, sampling_rate)) / units.mV, label='measurement')
                    ax[iCh][0].plot(np.abs(fft.time2freq(np.roll(analytic_traces[iCh], pos), sampling_rate)) / units.mV, '--', label='fit')
                    ax[iCh][0].set_xlim([0, 600])
                    ax[iCh][1].set_xlim([imin - 500, imax + 500])
                    ax[iCh][1].axvline(imin, linestyle='--', alpha=.8)
                    ax[iCh][1].axvline(imax, linestyle='--', alpha=.8)
            if(debug_obj and self.i_slope_fit_iterations % 50 == 0):
                sim_channel = station.get_sim_station().get_channel(0)[0]
                ax[4][0].plot(sim_channel.get_frequencies() / units.MHz, np.abs(pulse.get_analytic_pulse_freq(ampTheta, slope, phase, len(sim_channel.get_times()), sim_channel.get_sampling_rate(), bandpass=bandpass, quadratic_term=second_order)), '--', color='orange')
                ax[4][0].plot(sim_channel.get_frequencies() / units.MHz, np.abs(station.get_sim_station().get_channel(0)[0].get_frequency_spectrum()[1]), color='blue')
                ax[4][1].plot(sim_channel.get_frequencies() / units.MHz, np.abs(pulse.get_analytic_pulse_freq(ampPhi, slope, phase, len(sim_channel.get_times()), sim_channel.get_sampling_rate(), bandpass=bandpass, quadratic_term=second_order)), '--', color='orange')
                ax[4][1].plot(sim_channel.get_frequencies() / units.MHz, np.abs(sim_channel.get_frequency_spectrum()[2]), color='blue')
                ax[4][0].set_xlim([20, 500])
                ax[4][1].set_xlim([20, 500])
                # ax[4][0].set_yscale('log')
                # ax[4][1].set_yscale('log')
            logger.debug("amp phi = {:.4g}, amp theta = {:.4g}, slope = {:.4g} chi2 = {:.8g}".format(ampPhi, ampTheta, slope, chi2))
            if(debug_obj and self.i_slope_fit_iterations % 50 == 0):
        #                 ax[0][0].set_title("ratio = {:.2f} ({:.2f}), slope = {:.2g}, phase = {:.0f} ({:.2f}), chi2 = {:.2g}".format(ratio, ratio2, slope, phase / units.deg, phase2, chi2))
                fig.tight_layout()
                plt.show()
                self.i_slope_fit_iterations = 0
            self.i_slope_fit_iterations += 1
            return chi2

        method = "Nelder-Mead"
        # method = "BFGS"
        options = {'maxiter': 1000,
                   'disp': True}

        res = opt.minimize(obj_xcorr, x0=[-1], method=method, options=options)
        logger.info("slope xcorr fit, slope = {:.3g} with fmin = {:.3f}".format(res.x[0], res.fun))
        # plot objective function
        phase = 0
        ratio = 0
        slope = res.x[0]
        if slope > 0 or slope < -50:  # sanity check
            slope = -1.9
        analytic_pulse_theta_freq = pulse.get_analytic_pulse_freq(ratio, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
        analytic_pulse_phi_freq = pulse.get_analytic_pulse_freq(1 - ratio, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
        analytic_pulse_theta = pulse.get_analytic_pulse(ratio, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
        analytic_pulse_phi = pulse.get_analytic_pulse(1 - ratio, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)

        n_channels = len(V_timedomain)
        analytic_traces = np.zeros((n_channels, n_samples_time))
        positions = np.zeros(n_channels, dtype=np.int)
        max_xcorrs = np.zeros(n_channels)
        for iCh, trace in enumerate(V_timedomain):
            analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([analytic_pulse_theta_freq, analytic_pulse_phi_freq]), axis=0)
            analytic_traces[iCh] = fft.freq2time(analytic_trace_fft, sampling_rate)
            xcorr = np.abs(hp.get_normalized_xcorr(trace, analytic_traces[iCh]))
            positions[iCh] = np.argmax(np.abs(xcorr)) + 1
            max_xcorrs[iCh] = xcorr.max()
        pos = positions[np.argmax(max_xcorrs)]
        for iCh, trace in enumerate(V_timedomain):
            analytic_traces[iCh] = np.roll(analytic_traces[iCh], pos)

        res_amp = opt.minimize(obj_amplitude, x0=[1.], args=(slope, phase, pos, 0), method=method, options=options)
        logger.info("amplitude fit, Aphi = {:.3g} with fmin = {:.5e}".format(res_amp.x[0], res_amp.fun))
        Aphi = res_amp.x[0]
        Atheta = 0
        res_amp = opt.minimize(obj_amplitude, x0=[res_amp.x[0], 0], args=(slope, phase, pos, 0), method=method, options=options)
        logger.info("amplitude fit, Aphi = {:.3g} Atheta = {:.3g} with fmin = {:.5e}".format(res_amp.x[0], res_amp.x[1], res_amp.fun))
        Aphi = res_amp.x[0]
        Atheta = res_amp.x[1]
        # counts number of iterations in the slope fit. Used so we do not need to show the plots every iteration
        self.i_slope_fit_iterations = 0
        res_amp_slope = opt.minimize(obj_amplitude_slope, x0=[res_amp.x[0], res_amp.x[1], slope], args=(phase, pos, 'hilbert', False),
                                     method=method, options=options)

        # calculate uncertainties
        def Wrapper(params):
            return obj_amplitude_slope(params, phase, pos, 0)

        try:
            cov = covariance(Wrapper, res_amp_slope.x, 0.5, fast=True)
        except:
            cov = np.zeros((3, 3))
        logger.info("slope fit, Aphi = {:.3g}+-{:.3g} Atheta = {:.3g}+-{:.3g}, slope = {:.3g}+-{:.3g} with fmin = {:.5e}".format(res_amp_slope.x[0], cov[0, 0] ** 0.5,
                                                                                                                                     res_amp_slope.x[1], cov[1, 1] ** 0.5,
                                                                                                             res_amp_slope.x[2], cov[2, 2] ** 0.5, res_amp_slope.fun))
#         print(res_amp_slope)
        logger.info("covariance matrix \n{}".format(cov))
        if(cov[0, 0] > 0 and cov[1, 1] > 0 and cov[2, 2] > 0):
            logger.info("correlation matrix \n{}".format(hp.covariance_to_correlation(cov)))
        Aphi = res_amp_slope.x[0]
        Atheta = res_amp_slope.x[1]
        slope = res_amp_slope.x[2]
        Aphi_error = cov[0, 0] ** 0.5
        Atheta_error = cov[1, 1] ** 0.5
        slope_error = cov[2, 2] ** 0.5

         # plot objective function
        if 0:
            fo, ao = plt.subplots(1, 1)
            ss = np.linspace(-6, -0, 100)
            oos = [obj_amplitude_slope([res_amp_slope.x[0], res_amp_slope.x[1], s], phase, pos) for s in ss]
            ao.plot(ss, oos)

            n = 10
            x = np.linspace(res_amp_slope.x[0] * 0.6, res_amp_slope.x[0] * 1.4, n)
            y = np.linspace(-5, -1, n)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    Z[i, j] = obj_amplitude_slope([X[i, j], X[i, j] * res_amp_slope.x[1] / res_amp_slope.x[0], Y[i, j]], phase, pos)

            fig, ax = plt.subplots(1, 1)
            ax.pcolor(X, Y, Z, cmap='viridis_r', vmin=res_amp_slope.fun, vmax=res_amp_slope.fun * 2)

        analytic_pulse_theta = pulse.get_analytic_pulse(Atheta, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
        analytic_pulse_phi = pulse.get_analytic_pulse(Aphi, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
        analytic_pulse_theta_freq = pulse.get_analytic_pulse_freq(Atheta, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
        analytic_pulse_phi_freq = pulse.get_analytic_pulse_freq(Aphi, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)

        analytic_pulse_theta = np.roll(analytic_pulse_theta, pos)
        analytic_pulse_phi = np.roll(analytic_pulse_phi, pos)
        station_trace = np.array([np.zeros_like(analytic_pulse_theta), analytic_pulse_theta, analytic_pulse_phi])

        electric_field = NuRadioReco.framework.electric_field.ElectricField(use_channels)
        electric_field.set_trace(station_trace, sampling_rate)
        energy_fluence = trace_utilities.get_electric_field_energy_fluence(electric_field.get_trace(), electric_field.get_times())
        electric_field.set_parameter(efp.signal_energy_fluence, energy_fluence)
        electric_field.set_parameter_error(efp.signal_energy_fluence, np.array([0, Atheta_error, Aphi_error]))
        electric_field.set_parameter(efp.cr_spectrum_slope, slope)
        electric_field.set_parameter(efp.zenith, zenith)
        electric_field.set_parameter(efp.azimuth, azimuth)
        # calculate high level parameters
        x = np.sign(Atheta) * np.abs(Atheta) ** 0.5
        y = np.sign(Aphi) * np.abs(Aphi) ** 0.5
        sx = Atheta_error * 0.5
        sy = Aphi_error * 0.5
        pol_angle = np.arctan2(abs(y), abs(x))
        pol_angle_error = 1. / (x ** 2 + y ** 2) * (y ** 2 * sx ** 2 + x ** 2 + sy ** 2) ** 0.5  # gaussian error propagation
        logger.info("polarization angle = {:.1f} +- {:.1f}".format(pol_angle / units.deg, pol_angle_error / units.deg))
        electric_field.set_parameter(efp.polarization_angle, pol_angle)
        electric_field.set_parameter_error(efp.polarization_angle, pol_angle_error)

        # compute expeted polarization
        site = det.get_site(station.get_id())
        exp_efield = hp.get_lorentzforce_vector(zenith, azimuth, hp.get_magnetic_field_vector(site))
        cs = coordinatesystems.cstrafo(zenith, azimuth, site=site)
        exp_efield_onsky = cs.transform_from_ground_to_onsky(exp_efield)
        exp_pol_angle = np.arctan2(exp_efield_onsky[2], exp_efield_onsky[1])
        logger.info("expected polarization angle = {:.1f}".format(exp_pol_angle / units.deg))
        electric_field.set_parameter(efp.polarization_angle_expectation, exp_pol_angle)
        res_amp_second_order = opt.minimize(obj_amplitude_second_order, x0=[res_amp_slope.x[0], res_amp_slope.x[1], 0], args=(slope, phase, pos, 'hilbert', False),
                                     method=method, options=options)
        second_order_correction = res_amp_second_order.x[2]
        electric_field.set_parameter(efp.cr_spectrum_quadratic_term, second_order_correction)
        # figure out the timing of the electric field
        voltages_from_efield = trace_utilities.get_channel_voltage_from_efield(station, electric_field, use_channels, det, zenith, azimuth, self.antenna_provider, False)
        correlation = np.zeros(voltages_from_efield.shape[1] + station.get_channel(use_channels[0]).get_trace().shape[0] - 1)
        channel_trace_start_times = []
        for channel_id in use_channels:
            channel_trace_start_times.append(station.get_channel(channel_id).get_trace_start_time())
        average_trace_start_time = np.average(channel_trace_start_times)
        for i_trace, v_trace in enumerate(voltages_from_efield):
            channel = station.get_channel(use_channels[i_trace])
            time_shift = geo_utl.get_time_delay_from_direction(zenith, azimuth, det.get_relative_position(station.get_id(), use_channels[i_trace])) - (channel.get_trace_start_time() - average_trace_start_time)
            voltage_trace = np.roll(np.copy(v_trace), int(time_shift * electric_field.get_sampling_rate()))
            correlation += signal.correlate(voltage_trace, channel.get_trace())
        toffset = (np.arange(0, correlation.shape[0]) - channel.get_trace().shape[0]) / electric_field.get_sampling_rate()
        electric_field.set_trace_start_time(-toffset[np.argmax(correlation)] + average_trace_start_time)
        station.add_electric_field(electric_field)

        if debug:
            analytic_traces = np.zeros((n_channels, n_samples_time))
            for iCh, trace in enumerate(V_timedomain):
                analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([analytic_pulse_theta_freq, analytic_pulse_phi_freq]), axis=0)
                analytic_traces[iCh] = fft.freq2time(analytic_trace_fft, electric_field.get_sampling_rate())
                analytic_traces[iCh] = np.roll(analytic_traces[iCh], pos)
            fig, (ax2, ax2f) = plt.subplots(2, 1, figsize=(10, 8))
            lw = 2

            times = station.get_times() / units.ns
            ax2.plot(times, station.get_trace()[1] / units.mV * units.m, "-C0", label="analytic eTheta", lw=lw)
            ax2.plot(times, station.get_trace()[2] / units.mV * units.m, "-C1", label="analytic ePhi", lw=lw)
            tmax = times[np.argmax(station.get_trace()[2])]
            ax2.set_xlim(tmax - 40, tmax + 50)

            ff = station.get_frequencies() / units.MHz
            df = ff[1] - ff[0]
            ax2f.plot(ff[ff < 600], np.abs(station.get_frequency_spectrum()[1][ff < 600]) / df / units.mV * units.m, "-C0", label="analytic eTheta", lw=lw)
            ax2f.plot(ff[ff < 600], np.abs(station.get_frequency_spectrum()[2][ff < 600]) / df / units.mV * units.m, "-C1", label="analytic ePhi", lw=lw)

            if station.has_sim_station():
                sim_station = station.get_sim_station()
                logger.debug("station start time {:.1f}ns, relativ sim station time = {:.1f}".format(station.get_trace_start_time(), sim_station.get_trace_start_time()))
                df = (sim_station.get_frequencies()[1] - sim_station.get_frequencies()[0]) / units.MHz
                c = station.get_trace()[2].max() / sim_station.get_trace()[2].max()
                c = 1.
                ffsim = sim_station.get_frequencies()
                mask = (ffsim > 100 * units.MHz) & (ffsim < 500 * units.MHz)
                result = poly.polyfit(ffsim[mask], np.log10(np.abs(sim_station.get_frequency_spectrum()[2][mask]) / df / units.mV * units.m), 1, full=True)
                logger.info("polyfit result = {:.2g}  {:.2g}".format(*result[0]))
                ax2.plot(sim_station.get_times() / units.ns, sim_station.get_trace()[1] / units.mV * units.m * c, "--C0", label="simulation eTheta", lw=lw)
                ax2.plot(sim_station.get_times() / units.ns, sim_station.get_trace()[2] / units.mV * units.m * c, "--C1", label="simulation ePhi", lw=lw)
                ax2f.plot(sim_station.get_frequencies() / units.MHz, np.abs(sim_station.get_frequency_spectrum()[1]) / df / units.mV * units.m * c, "--C0", label="simulation eTheta", lw=lw)
                ax2f.plot(sim_station.get_frequencies() / units.MHz, np.abs(sim_station.get_frequency_spectrum()[2]) / df / units.mV * units.m * c, "--C1", label="simulation ePhi", lw=lw)

                ax2f.plot(ffsim / units.MHz, 10 ** (result[0][0] + result[0][1] * ffsim), "C3:")

            ax2.legend(fontsize="xx-small")
            ax2.set_xlabel("time [ns]")
            ax2.set_ylabel("electric-field [mV/m]")
            ax2f.set_ylim(1e-3, 5)
            ax2f.set_xlabel("Frequency [MHz]")
            ax2f.set_xlim(100, 500)
            ax2f.semilogy(True)
            if sim_present:
                sim = station.get_sim_station()
                fig.suptitle("Simulation: Zenith {:.1f}, Azimuth {:.1f}".format(np.rad2deg(sim[stnp.zenith]), np.rad2deg(sim[stnp.azimuth])))
            else:
                fig.suptitle("Data: reconstructed zenith {:.1f}, azimuth {:.1f}".format(np.rad2deg(zenith), np.rad2deg(azimuth)))
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)
            if(debug_plotpath is not None):
                fig.savefig(os.path.join(debug_plotpath, 'run_{:05d}_event_{:06d}_efield.png'.format(evt.get_run_number(), evt.get_id())))
                plt.close(fig)

            # plot antenna response and channels
            fig, ax = plt.subplots(len(V), 3, sharex='col', sharey='col')
            for iCh in range(len(V)):
                mask = ff > 100
                ax[iCh, 0].plot(ff[mask], np.abs(efield_antenna_factor[iCh][0])[mask], label="theta, channel {}".format(use_channels[iCh]), lw=lw)
                ax[iCh, 0].plot(ff[mask], np.abs(efield_antenna_factor[iCh][1])[mask], label="phi, channel {}".format(use_channels[iCh]), lw=lw)
                ax[iCh, 0].legend(fontsize='xx-small')
                ax[iCh, 0].set_xlim(100, 500)
                ax[iCh, 1].set_xlim(400, 600)
                ax[iCh, 2].set_xlim(400, 600)
                ax[iCh, 1].plot(times, V_timedomain[iCh] / units.micro / units.V, lw=lw)
                ax[iCh, 1].plot(times, analytic_traces[iCh] / units.micro / units.V, '--', lw=lw)
                ax[iCh, 2].plot(times, (V_timedomain[iCh] - analytic_traces[iCh]) / units.micro / units.V, '-', lw=lw)
                ax[iCh, 0].set_ylabel("H [m]")
                ax[iCh, 1].set_ylabel(r"V [$\mu$V]")
                ax[iCh, 2].set_ylabel(r"$\Delta$V [$\mu$V]")
                RMS = det.get_noise_RMS(station.get_id(), 0)
                ax[iCh, 1].text(0.6, 0.8, 'S/N={:.1f}'.format(np.max(np.abs(V_timedomain[iCh])) / RMS), transform=ax[iCh, 1].transAxes)
            ax[0][2].set_ylim(ax[0][1].get_ylim())
            ax[-1, 1].set_xlabel("time [ns]")
            ax[-1, 2].set_xlabel("time [ns]")
            ax[-1, 0].set_xlabel("frequency [MHz]")
            fig.tight_layout()
            if(debug_plotpath is not None):
                fig.savefig(os.path.join(debug_plotpath, 'run_{:05d}_event_{:06d}_channels.png'.format(evt.get_run_number(), evt.get_id())))
                plt.close(fig)

    def end(self):
        pass
