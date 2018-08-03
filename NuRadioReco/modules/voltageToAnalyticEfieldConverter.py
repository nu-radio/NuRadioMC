from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from NuRadioReco.detector import detector
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from scipy import signal
from NuRadioReco.detector import antennapattern
from radiotools import plthelpers as php
from numpy.polynomial import polynomial as poly
from scipy.signal import correlate
from scipy import optimize as opt
import os
from NuRadioReco.utilities import analytic_pulse as pulse
from NuRadioReco.modules.voltageToEfieldConverter import get_array_of_channels
import time
import matplotlib.pyplot as plt
from radiotools import helper as hp
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

    def run(self, evt, station, det, debug=False, debug_plotpath=None,
            use_channels=[0, 1, 2, 3],
            bandpass=[100 * units.MHz, 500 * units.MHz],
            useMCdirection=False):
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
        bandpass: [float, float]
            the lower and upper frequecy for which the analytic pulse is calculated.
            A butterworth filter of 10th order and a rectangular filter is applied.
        """
        self.__counter += 1
        event_time = station.get_station_time()
        station_id = station.get_id()
        logger.info("event {}, station {}".format(evt.get_id(), station_id))
        if useMCdirection and (station.get_sim_station() is not None):
            zenith = station.get_sim_station()['zenith']
            azimuth = station.get_sim_station()['azimuth']
            sim_present = True
        else:
            logger.warning("Using reconstructed angles as no simulation present")
            zenith = station['zenith']
            azimuth = station['azimuth']
            sim_present = False

        channels = station.get_channels()
        efield_antenna_factor, V, V_timedomain = get_array_of_channels(station, use_channels,
                                                                       det, zenith, azimuth, self.antenna_provider,
                                                                       time_domain=True)
        sampling_rate = channels[0].get_sampling_rate()
        n_samples_time = V_timedomain.shape[1]

        debug_obj = 0
        noise_RMS = det.get_noise_RMS(station.get_id(), 0)

#         V_timedomain /= (units.micro * units.V)

        def obj(params):
            theta_amp_p1 = 0
            phi_amp_p1 = 0
            theta_phase_p0 = 0
            phi_phase_p0 = 0
            if(len(params) == 2):
                theta_amp_p0, phi_amp_p0 = params
            elif(len(params) == 4):
                theta_amp_p0, phi_amp_p0, theta_amp_p1, phi_amp_p1 = params
            elif(len(params) == 6):
                theta_amp_p0, phi_amp_p0, theta_amp_p1, phi_amp_p1, theta_phase_p0, phi_phase_p0 = params
#             theta_amp_p0 -= 6
#             phi_amp_p0 -= 6

            analytic_pulse_theta = pulse.get_analytic_pulse_freq(theta_amp_p0, theta_amp_p1, theta_phase_p0, n_samples_time, sampling_rate, bandpass=bandpass)
            analytic_pulse_phi = pulse.get_analytic_pulse_freq(phi_amp_p0, phi_amp_p1, phi_phase_p0, n_samples_time, sampling_rate, bandpass=bandpass)
            chi2 = 0
            # first determine the time offset of the analytic pulse
            # use time offset of channel with the best xcorr

            if(debug_obj):
                fig, ax = plt.subplots(4, 1, sharex=True)

            n_channels = len(V_timedomain)
            analytic_traces = np.zeros((n_channels, n_samples_time))
            positions = np.zeros(n_channels, dtype=np.int)
            max_xcorrs = np.zeros(n_channels)
            for iCh, trace in enumerate(V_timedomain):
                analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([analytic_pulse_theta, analytic_pulse_phi]), axis=0)
                analytic_traces[iCh] = np.fft.irfft(analytic_trace_fft, norm='ortho') / 2 ** 0.5
                xcorr = np.abs(hp.get_normalized_xcorr(trace, analytic_traces[iCh]))
                positions[iCh] = np.argmax(np.abs(xcorr)) + 1
                max_xcorrs[iCh] = xcorr.max()
            pos = positions[np.argmax(max_xcorrs)]

            for iCh, trace in enumerate(V_timedomain):
                tmp = np.sum(np.abs(trace - np.roll(analytic_traces[iCh], pos)))
                if(debug_obj):
                    ax[iCh].plot(trace, label='measurement')
                    ax[iCh].plot(np.roll(analytic_traces[iCh], pos), '--', label='fit')
#                 logger.debug("channel {:d}: optimal position {:d}, chi2 = {:4g}".format(iCh, pos, tmp))
                chi2 += tmp
            if(debug_obj):
                ax[0].set_title("Atheta = {:.2g}, Aphi = {:.2g}, chi2 = {:.4g}".format(theta_amp_p0, phi_amp_p0, chi2))
                fig.tight_layout()
                plt.show()
            return chi2

        def obj_xcorr(params):
#             if(len(params) == 3):
#                 slope, ratio2, phase2 = params
#             elif(len(params) == 2):
#                 slope, phase2 = params
#                 ratio2 = -1000
#             elif(len(params) == 1):
#                 phase2 = 0
#                 ratio2 = -1000
#                 slope = params[0]

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

            if(debug_obj):
                fig, ax = plt.subplots(4, 2, sharex='col')

            n_channels = len(V_timedomain)
            analytic_traces = np.zeros((n_channels, n_samples_time))
            positions = np.zeros(n_channels, dtype=np.int)
            max_xcorrs = np.zeros(n_channels)
            # first determine the position with the larges xcorr
            for iCh, trace in enumerate(V_timedomain):
                analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([analytic_pulse_theta, analytic_pulse_phi]), axis=0)
                analytic_traces[iCh] = np.fft.irfft(analytic_trace_fft, norm='ortho') / 2 ** 0.5
                xcorr = np.abs(hp.get_normalized_xcorr(trace, analytic_traces[iCh]))
#                 pos = np.argmax(np.abs(xcorr)) + 1
                positions[iCh] = np.argmax(np.abs(xcorr)) + 1
                max_xcorrs[iCh] = xcorr.max()
                chi2 -= xcorr.max()
#             pos = positions[np.argmax(max_xcorrs)]
#             # calculate chi2 by summing all xcorrs at the same position
#             for iCh, trace in enumerate(V_timedomain):
#                 xcorr = hp.get_normalized_xcorr(trace, analytic_traces[iCh])
#                 chi2 -= np.abs(xcorr[pos])
#                 if(debug_obj):
#                     ax[iCh, 0].plot(trace, label='measurement')
#                     trace_ana = np.roll(analytic_traces[iCh], pos) / analytic_traces[iCh].max() * trace.max() * np.sign(xcorr[pos])
#                     ax[iCh, 0].plot(trace_ana, '--', label='fit')
#                     ax[iCh, 0].set_xlim(1600, 2400)
#                     specana = np.fft.rfft(trace_ana, norm='ortho') * 2 ** 0.5
#                     specV = np.fft.rfft(trace, norm='ortho') * 2 ** 0.5
#                     ax[iCh, 1].plot(np.abs(specV))
#                     ax[iCh, 1].plot(np.abs(specana), '--')
            if(debug_obj):
                ax[0, 0].set_title("ratio = {:.2f}, slope = {:.4g}, phase = {:.0f} ({:.4f}), chi2 = {:.4g}".format(ratio, slope, phase / units.deg, phase2, chi2))
                fig.tight_layout()
                plt.show()
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
                analytic_traces[iCh] = np.fft.irfft(analytic_trace_fft, norm='ortho') / 2 ** 0.5

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

        def obj_amplitude_slope(params, phase, pos, debug_obj=0):
            ampPhi, ampTheta, slope = params

            analytic_pulse_theta = pulse.get_analytic_pulse_freq(ampTheta, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
            analytic_pulse_phi = pulse.get_analytic_pulse_freq(ampPhi, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
            chi2 = 0

            if(debug_obj):
                fig, ax = plt.subplots(4, 2, sharex=True)

            n_channels = len(V_timedomain)
            analytic_traces = np.zeros((n_channels, n_samples_time))
            # first determine the position with the larges xcorr
            for iCh, trace in enumerate(V_timedomain):
                argmax = np.argmax(np.abs(trace))
                imin = np.int(argmax - 50 * sampling_rate)
                imax = np.int(argmax + 50 * sampling_rate)

                analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([analytic_pulse_theta, analytic_pulse_phi]), axis=0)
                analytic_traces[iCh] = np.fft.irfft(analytic_trace_fft, norm='ortho') / 2 ** 0.5
                tmp = np.sum(np.abs(trace[imin:imax] - np.roll(analytic_traces[iCh], pos)[imin:imax]) / noise_RMS)
                chi2 += tmp ** 2
                if(debug_obj):
                    ax[iCh][0].plot(trace, label='measurement')
                    ax[iCh][0].plot(np.roll(analytic_traces[iCh], pos), '--', label='fit')
                    ax[iCh][1].plot(trace - np.roll(analytic_traces[iCh], pos), label='delta')
            logger.debug("amp phi = {:.4g}, amp theta = {:.4g}, slope = {:.4g} chi2 = {:.8g}".format(ampPhi, ampTheta, slope, chi2))
            if(debug_obj):
#                 ax[0][0].set_title("ratio = {:.2f} ({:.2f}), slope = {:.2g}, phase = {:.0f} ({:.2f}), chi2 = {:.2g}".format(ratio, ratio2, slope, phase / units.deg, phase2, chi2))
                fig.tight_layout()
                plt.show()
            return chi2

        method = "Nelder-Mead"
        # method = "BFGS"
        options = {'maxiter': 1000,
                   'disp': True}

        if 0:  # test first method
            res = opt.minimize(obj, x0=[1, 1, -1, -1, 0, 0], method=method, options=options)
            logger.debug(res)
    #         res = opt.minimize(obj, x0=[res.x[0], res.x[1], -1, -1], method=method, options=options)
    #         print(res)
    #         res = opt.minimize(obj, x0=[res.x[0], res.x[1], res.x[2], res.x[3], 0, 0], method=method, options=options)
            logger.debug(res)
            analytic_pulse_theta_freq = pulse.get_analytic_pulse_freq(res.x[0], res.x[2], res.x[4], n_samples_time, sampling_rate, bandpass=bandpass)
            analytic_pulse_phi_freq = pulse.get_analytic_pulse_freq(res.x[1], res.x[3], res.x[5], n_samples_time, sampling_rate, bandpass=bandpass)
            analytic_pulse_theta = pulse.get_analytic_pulse(res.x[0], res.x[2], res.x[4], n_samples_time, sampling_rate, bandpass=bandpass)
            analytic_pulse_phi = pulse.get_analytic_pulse(res.x[1], res.x[3], res.x[5], n_samples_time, sampling_rate, bandpass=bandpass)

#         res = opt.minimize(obj_xcorr, x0=[0.5, 0, 1], method=method, options=options)
#         phase = np.arctan(res.x[2])  # project -inf..+inf to -0.5 pi..0.5 pi

        res = opt.minimize(obj_xcorr, x0=[-1], method=method, options=options)
        logger.info("slope xcorr fit, slope = {:.3g} with fmin = {:.3f}".format(res.x[0], res.fun))
        # plot objective function
        if 0:
            fo, ao = plt.subplots(1, 1)
            ss = np.linspace(-6, -1, 100)
            oos = [obj_xcorr([s]) for s in ss]
            ao.plot(ss, oos)
#             plt.show()
#         ratio = 0.5
        phase = 0
#         res = opt.minimize(obj_xcorr, x0=[res.x[0], -1], method=method, options=options)
        ratio = 0
#         phase = np.arctan(res.x[1])  # project -inf..+inf to -0.5 pi..0.5 pi
        slope = res.x[0]
#         res = opt.minimize(obj_xcorr, x0=[res.x[0], -100], method=method, options=options)
#         ratio = (np.arctan(res.x[1]) + np.pi * 0.5) / np.pi  # project -inf..inf on 0..1
#         phase = np.arctan(0)  # project -inf..+inf to -0.5 pi..0.5 pi
#         slope = res.x[0]
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
            analytic_traces[iCh] = np.fft.irfft(analytic_trace_fft, norm='ortho') / 2 ** 0.5
            xcorr = np.abs(hp.get_normalized_xcorr(trace, analytic_traces[iCh]))
            positions[iCh] = np.argmax(np.abs(xcorr)) + 1
            max_xcorrs[iCh] = xcorr.max()
        pos = positions[np.argmax(max_xcorrs)]
        for iCh, trace in enumerate(V_timedomain):
            analytic_traces[iCh] = np.roll(analytic_traces[iCh], pos)

        res_amp = opt.minimize(obj_amplitude, x0=[1.], args=(-1.9, phase, pos, 0), method=method, options=options)
        logger.info("amplitude fit, Aphi = {:.3g} with fmin = {:.5e}".format(res_amp.x[0], res_amp.fun))
        Aphi = res_amp.x[0]
        Atheta = 0
        res_amp = opt.minimize(obj_amplitude, x0=[res_amp.x[0], 0], args=(-1.9, phase, pos, 0), method=method, options=options)
        logger.info("amplitude fit, Aphi = {:.3g} Atheta = {:.3g} with fmin = {:.5e}".format(res_amp.x[0], res_amp.x[1], res_amp.fun))
        Aphi = res_amp.x[0]
        Atheta = res_amp.x[1]

        res_amp_slope = opt.minimize(obj_amplitude_slope, x0=[res_amp.x[0], res_amp.x[1], -1.9], args=(phase, pos),
                                     method=method, options=options)

        # calculate uncertainties
        def Wrapper(params):
            return obj_amplitude_slope(params, phase, pos, 0)

        try:
            cov = covariance(Wrapper, res_amp_slope.x, 0.5, fast=True)
        except:
            cov = np.zeros((3, 3))
        logger.info("amplitude fit, Aphi = {:.3g}+-{:.3g} Atheta = {:.3g}+-{:.3g}, slope = {:.3g}+-{:.3g} with fmin = {:.5e}".format(res_amp_slope.x[0], cov[0, 0] ** 0.5,
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
        station.set_parameter("signal_energy_fluence", np.array([0, Atheta, Aphi]))
        station.set_parameter_error("signal_energy_fluence", np.array([0, Atheta_error, Aphi_error]))

#         cov = covariance(Wrapper, res_amp_slope.x, 0.5, fast=False)
#         print(cov)

#         res_amp_slope2 = opt.basinhopping(obj_amplitude_slope, x0=[res_amp_slope.x[0], res_amp_slope.x[1], -2],
#                                           minimizer_kwargs={'args':(phase, pos)}, disp=True)
#         logger.info("amplitude fit, Aphi = {:.3g} Atheta = {:.3g}, slope = {:.3g} with fmin = {:.5e}".format(res_amp_slope2.x[0], res_amp_slope2.x[1],
#                                                                                                              res_amp_slope2.x[2], res_amp_slope2.fun))

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

#             plt.show()

        analytic_pulse_theta = pulse.get_analytic_pulse(Atheta, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
        analytic_pulse_phi = pulse.get_analytic_pulse(Aphi, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
        analytic_pulse_theta_freq = pulse.get_analytic_pulse_freq(Atheta, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)
        analytic_pulse_phi_freq = pulse.get_analytic_pulse_freq(Aphi, slope, phase, n_samples_time, sampling_rate, bandpass=bandpass)

#         print("Aphi = {:.4g}".format(Aphi * conversion_factor_integrated_signal * 1e-12))
#         print('fphi spec {:.4e}'.format(np.sum(np.abs(analytic_pulse_phi_freq) ** 2) / sampling_rate * conversion_factor_integrated_signal * 1e-12))
#         print('fphi time {:.4e}'.format(np.sum(np.abs(analytic_pulse_phi) ** 2) / sampling_rate * conversion_factor_integrated_signal * 1e-12))

        analytic_pulse_theta = np.roll(analytic_pulse_theta, pos)
        analytic_pulse_phi = np.roll(analytic_pulse_phi, pos)
        station_trace = np.array([np.zeros_like(analytic_pulse_theta), analytic_pulse_theta, analytic_pulse_phi])
        station.set_trace(station_trace, sampling_rate)

        # calculate high level parameters
        x = np.sign(Atheta) * np.abs(Atheta) ** 0.5
        y = np.sign(Aphi) * np.abs(Aphi) ** 0.5
        sx = Atheta_error * 0.5
        sy = Aphi_error * 0.5
        pol_angle = np.arctan2(y, x)
        pol_angle_error = 1. / (x ** 2 + y ** 2) * (y ** 2 * sx ** 2 + x ** 2 + sy ** 2) ** 0.5  # gaussian error propagation
        logger.info("polarization angle = {:.1f} +- {:.1f}".format(pol_angle / units.deg, pol_angle_error / units.deg))
        station.set_parameter('polarization_angle', pol_angle)
        station.set_parameter_error('polarization_angle', pol_angle_error)

        if debug:
            analytic_traces = np.zeros((n_channels, n_samples_time))
            for iCh, trace in enumerate(V_timedomain):
                analytic_trace_fft = np.sum(efield_antenna_factor[iCh] * np.array([analytic_pulse_theta_freq, analytic_pulse_phi_freq]), axis=0)
                analytic_traces[iCh] = np.fft.irfft(analytic_trace_fft, norm='ortho') / 2 ** 0.5
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
                # ax2.plot(times_sim / units.ns, efield_sim[0] / units.mV * units.m, "--", label="simulation eR")
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

#             ax2f.set_ylim(1e-3)
            ax2.legend(fontsize="xx-small")
            ax2.set_xlabel("time [ns]")
            ax2.set_ylabel("electric-field [mV/m]")
#             axf.set_xlabel("Frequency [MHz]")
            ax2f.set_ylim(1e-3, 5)
            ax2f.set_xlabel("Frequency [MHz]")
            ax2f.set_xlim(100, 500)
            ax2f.semilogy(True)
            if sim_present:
                sim = station.get_sim_station()
                fig.suptitle("Simulation: Zenith {:.1f}, Azimuth {:.1f}".format(np.rad2deg(sim["zenith"]), np.rad2deg(sim["azimuth"])))
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

#         p_mag_pre = np.zeros((3, 2))
#         p_mag_stats_pre = []
#         p_phase_pre = np.zeros((3, 2))
#         p_phase_stats_pre = []
#         if station.has_sim_station():
#             sim_station = station.get_sim_station()
#             max_bin = np.argmax(sim_station.get_hilbert_envelope_mag())
#             sim_trace = np.roll(sim_station.get_trace(), -max_bin, axis=1)
#             sim_times = sim_station.get_times()
#             sim_ff = sim_station.get_frequencies()
#             sim_spectrum = np.fft.rfft(sim_trace, norm='ortho', axis=-1) * 2 ** 0.5
#             mask = (sim_ff >= 100 * units.MHz) & (sim_ff <= 500 * units.MHz)
#
#             for iPol in xrange(0, 3):
#                 p0_mag, p0_mag_stats = poly.polyfit(sim_ff[mask], np.log10(np.abs(sim_spectrum[iPol][mask])),
#                                                     1, w=1. / np.log10(np.abs(sim_spectrum[iPol][mask])), full=True)
#                 p_mag_pre[iPol] = p0_mag
#                 p_mag_stats_pre.append(p0_mag_stats)
#                 p0_phase, p0_phase_stats = poly.polyfit(sim_ff[mask], np.unwrap(np.angle(sim_spectrum[iPol][mask])),
#                                                         1, full=True)
#                 p_phase_pre[iPol] = p0_phase
#                 p_phase_stats_pre.append(p0_phase_stats)
#
#             if 0:
#                 fsim, (axtrace, axmag, axphase) = plt.subplots(1, 3, figsize=(15, 6))
#                 axtrace.plot(sim_times, sim_trace[0], "-{}".format(php.get_color(0)))
#                 axtrace.plot(sim_times, sim_trace[1], "-{}".format(php.get_color(1)))
#                 axtrace.plot(sim_times, sim_trace[2], "-{}".format(php.get_color(2)))
#     #             axtrace.plot(sim_times, sim_station.get_trace()[0], "--{}".format(php.get_color(0)))
#     #             axtrace.plot(sim_times, sim_station.get_trace()[1], "--{}".format(php.get_color(1)))
#     #             axtrace.plot(sim_times, sim_station.get_trace()[2], "--{}".format(php.get_color(2)))
#                 xxx = np.linspace(100 * units.MHz, 500 * units.MHz)
#                 for iPol in xrange(0, 3):
#                     axmag.plot(sim_ff / units.MHz, np.abs(sim_spectrum[iPol]), ".{}".format(php.get_color(iPol)))
#                     axphase.plot(sim_ff[mask] / units.MHz, np.rad2deg(np.unwrap(np.angle(sim_spectrum[iPol][mask]))), ".{}".format(php.get_color(iPol)))
#                     axmag.plot(xxx / units.MHz, 10 ** poly.polyval(xxx, p_mag_pre[iPol]), "-{}".format(php.get_color(iPol)))
#                     axphase.plot(xxx / units.MHz, np.rad2deg(poly.polyval(xxx, p_phase_pre[iPol])), "-{}".format(php.get_color(iPol)))
#                 axmag.set_xlim(100, 500)
#                 axphase.set_xlim(100, 500)
#                 plt.tight_layout()
#
#         def get_voltage_spectrum(params, frequencies):
#             amp_p0_theta, amp_p1_theta, phase_p0_theta, phase_p1_theta, amp_p0_phi, amp_p1_phi, phase_p0_phi, phase_p1_phi = params
#             efield_trace_theta_analytic = get_analytic_pulse_freq(amp_p0_theta, amp_p1_theta, phase_p0_theta,
#                                                                   phase_p1_theta, frequencies)
#             efield_trace_phi_analytic = get_analytic_pulse_freq(amp_p0_phi, amp_p1_phi, phase_p0_phi, phase_p1_phi, frequencies)
#
#             voltage_spectrum_analytic = np.einsum('ij..., j...', efield_antenna_factor,
#                                                   np.array([efield_trace_theta_analytic, efield_trace_phi_analytic])).T
#             return voltage_spectrum_analytic
#
#         def get_voltage_traces(params, frequencies):
#             return np.fft.irfft(get_voltage_spectrum(params, frequencies),
#                                 norm="ortho", axis=-1) / 2 ** 0.5
#
#         def obj(params, voltage_traces, frequencies):
#             voltage_traces_analytic = get_voltage_traces(params, frequencies)
#             chi2 = 0
# #                                                      np.array([efield_trace_theta_analytic, efield_trace_phi_analytic]))
#             for iT, trace_analytic in enumerate(voltage_traces_analytic):
#                 chi2 += np.sum(np.abs(voltage_traces[iT] - trace_analytic) ** 2 / np.abs(voltage_traces).max())
#             return chi2
#
#         def get_voltage_spectrum_mag(params, frequencies):
#             amp_p0_theta, amp_p1_theta, phase_offset_theta, amp_p0_phi, amp_p1_phi, phase_offset_phi = params
# #             t = (2 * ((phase_offset_theta - 0) / (2 * np.pi)) - 1)
# #             t = min(1, t)
# #             t = max(-1, t)
# #             phase_offset_theta_internal = np.arcsin(t)
# #             print(phase_offset_theta_internal)
# #             t = 2 * ((phase_offset_phi - 0) / (2 * np.pi)) - 1
# #             t = min(1, t)
# #             t = max(-1, t)
# #             phase_offset_phi_internal = np.arcsin(t)
# #             print(phase_offset_phi_internal)
#             efield_trace_theta_analytic = get_analytic_pulse_freq(amp_p0_theta, amp_p1_theta, phase_offset_theta, 0, frequencies)
#             efield_trace_phi_analytic = get_analytic_pulse_freq(amp_p0_phi, amp_p1_phi, phase_offset_phi, 0, frequencies)
#
#             voltage_spectrum_analytic = np.einsum('ij..., j...', efield_antenna_factor,
#                                                   np.array([efield_trace_theta_analytic, efield_trace_phi_analytic])).T
#
# #             voltage_spectrum_analytic2 = np.zeros_like(voltage_spectrum_analytic)
# #             for iCh in xrange(4):
# #                 voltage_spectrum_analytic2[iCh] = efield_antenna_factor[iCh][0] * efield_trace_theta_analytic + efield_antenna_factor[iCh][1] * efield_trace_phi_analytic
#
#             return voltage_spectrum_analytic
#
#         def get_voltage_traces_mag(params, frequencies):
#             return np.fft.irfft(get_voltage_spectrum_mag(params, frequencies),
#                                 norm="ortho", axis=-1) / 2 ** 0.5
#
#         def get_shift(params, voltage_traces, voltage_traces_analytic):
#             corr = np.zeros_like(voltage_traces_analytic)
#             for iT, trace_analytic in enumerate(voltage_traces_analytic):
#                 corr[iT] = signal.correlate(voltage_traces[iT], voltage_traces_analytic[iT], mode='same')
#             corr_mag = np.linalg.norm(corr, axis=0)
#             max_bin = np.argmax(corr_mag)
#             n_samples = len(corr_mag)
#             i_shift = max_bin - n_samples // 2
#             return i_shift
#
#         def obj_mag(params, voltage_traces, frequencies):
#             voltage_traces_analytic = get_voltage_traces_mag(params, frequencies)
#             i_shift = get_shift(params, voltage_traces, voltage_traces_analytic)
#
#             chi2 = 0
#             for iT, trace_analytic in enumerate(voltage_traces_analytic):
#                 chi2 += np.sum((voltage_traces[iT] - np.roll(trace_analytic, i_shift)) ** 2 / np.abs(voltage_traces).max() ** 2)
#
# #             print("{:.2g}".format(chi2), params)
#             if 0:
#                 fig, ax = plt.subplots(2, 1, sharex=True)
#                 for iT, trace_analytic in enumerate(voltage_traces_analytic):
#                     corr = signal.correlate(voltage_traces[iT], voltage_traces_analytic[iT], mode='same')
#     #                 chi2 += np.sum(np.abs(voltage_traces[iT] - trace_analytic) ** 2 / np.abs(voltage_traces).max())
#                     ax[0].plot(voltage_traces[iT], color=php.get_color(iT))
#                     ax[0].plot(np.roll(trace_analytic, i_shift), '--', color=php.get_color(iT))
#                     ax[1].plot(corr, color=php.get_color(iT))
#                 ax[0].set_title('correlation')
#                 plt.tight_layout()
#                 plt.show()
#             return chi2
#
#         # compute the time shift due to the antenna model
# #         n_samples = len(V_timedomain[0])
# #         trace_delta = np.zeros(n_samples)
# #         i_middle = n_samples // 2
# #         trace_delta[i_middle] = 1
# #         spec_delta = np.fft.rfft(trace_delta, norm='ortho')
# #         mask = (frequencies < 100 * units.MHz) | (frequencies > 500 * units.MHz)
# #         spec_delta[mask] = 0
# #         V_delta_spec = np.einsum('ij..., j...', efield_antenna_factor, np.array([spec_delta, np.zeros_like(spec_delta, dtype=np.complex)])).T
# #         V_delta = np.fft.irfft(V_delta_spec, norm='ortho', axis=-1)
# #         print('middle of trace is ', i_middle)
# #         fig_delta, ax_delta = plt.subplots(1, 1)
# #         for i, V_temp in enumerate(V_delta):
# #             print(i, np.argmax(np.abs(V_temp)))
# #             from scipy import signal
# #             ax_delta.plot(times, V_temp)
# #             ax_delta.plot(times, np.abs(signal.hilbert(V_temp)), '--')
# #         plt.show()
#
#         method = "Nelder-Mead"
#         # method = "BFGS"
#         options = {'maxiter': 1000,
#                    'disp': True}
#         x0 = [p_mag_pre[1][0], p_mag_pre[1][1], p_phase_pre[1][0], p_phase_pre[1][1],
#               p_mag_pre[2][0], p_mag_pre[2][1], p_phase_pre[2][0], p_phase_pre[2][1]]
# #         plt.close("all")
# #         res = opt.minimize(obj, x0=x0, args=(V_timedomain, frequencies), method=method, options=options)
#         x0_mag = [p_mag_pre[1][0], p_mag_pre[1][1], np.pi,
#                   p_mag_pre[2][0], p_mag_pre[2][1], np.pi]
#         t = time.time()
#         res_mag = opt.minimize(obj_mag, x0=x0_mag, args=(V_timedomain, frequencies), method=method, options=options)
#         print("fit needed {:.1f} s".format(time.time() - t))
#         print(res_mag.x)
#         voltage_traces_analytic = get_voltage_traces_mag(res_mag.x, frequencies)
#         i_shift = get_shift(res_mag.x, V_timedomain, voltage_traces_analytic)
# #         P = ((2 * np.pi) / (2.))(np.sin(P) + 1)
# #         res = res_mag
#
#         for i, trace in enumerate(voltage_traces_analytic):
#             ax_V.plot(times / units.ns, np.roll(trace, i_shift) / units.mV, "C{}--".format(i))
#             ax_V_res.plot(times / units.ns, (np.roll(trace, i_shift) - V_timedomain[i]) / units.mV, "C{}--".format(i))
#         ax_V.legend(fontsize='small')
#         ax_V_res.set_xlabel("time [ns]")
#         ax_V.set_ylabel("voltage [mV]")
#         ax_V_res.set_ylabel(r"$\Delta$")
#         ax_V.set_title("S/N {rms0:.1f}, {rms1:.1f}, {rms2:.1f}, {rms3:.1f}".format(rms0=channels.values()[0]['SNR'],
#                                                                                          rms1=channels.values()[1]['SNR'],
#                                                                                         rms2=channels.values()[2]['SNR'],
#                                                                                          rms3=channels.values()[3]['SNR']))
#         fig_V.tight_layout()
# #         fig_analytic, ax_analytic = plt.subplots(1, 1)
# #         yy = get_analytic_pulse(res_mag.x[0], res_mag.x[1], res_mag.x[2], 0, frequencies)
# #         ax_analytic.plot(times, yy / units.mV * units.m, label='eTheta')
# #         yy = get_analytic_pulse(res_mag.x[3], res_mag.x[4], res_mag.x[5], 0, frequencies)
# #         ax_analytic.plot(times, yy / units.mV * units.m, label='ePhi')
# #         ax_analytic.legend(fontsize='small')
# #         plt.show()
#
#         # solve it in a vectorized way
#         efield3_f = np.moveaxis(stacked_lstsq(np.moveaxis(efield_antenna_factor, 2, 0), np.moveaxis(V, 1, 0)), 0, 1)
#         # add eR direction
#         efield3_f = np.array([np.zeros_like(efield3_f[0], dtype=np.complex),
#                              efield3_f[0],
#                              efield3_f[1]])
#
#         station.set_frequency_spectrum(efield3_f, channels.values()[0].get_sampling_rate())
#

#             fig_E, (ax_E, ax_E_res) = plt.subplots(2, 1, sharex=True)
#             efield3 = np.fft.irfft(efield3_f, norm="ortho") / 2 ** 0.5
# #             efield = np.fft.irfft(efield_f, norm="ortho") / 2 ** 0.5
# #             efield2 = np.fft.irfft(efield_f2, norm="ortho") / 2 ** 0.5
# #             efield21 = np.fft.irfft(E1, norm="ortho") / 2 ** 0.5
# #             efield22 = np.fft.irfft(E2, norm="ortho") / 2 ** 0.5
# #             efield31 = np.fft.irfft(E3, norm="ortho") / 2 ** 0.5
#
#             times = station.get_times() / units.ns
# #             ax2.plot(times, efield21 / units.mV * units.m, ":C2", label="exact solution Ch 0+1")
# #             ax2.plot(times, efield31 / units.mV * units.m, ":C3", label="exact solution Ch 2+3")
# #             ax2.plot(times, efield3[1] / units.mV * units.m, "-C0", label="4 stations lsqr eTheta")
# #             ax2.plot(times, efield3[2] / units.mV * units.m, "-C1", label="4 stations lsqr ePhi")
#
#             etheta_analytic = np.roll(get_analytic_pulse(res_mag.x[0], res_mag.x[1], res_mag.x[2], 0, frequencies), i_shift)
#             ax2.plot(times, etheta_analytic / units.mV * units.m, "--C0", label="analytic eTheta")
#             ephi_analytic = np.roll(get_analytic_pulse(res_mag.x[3], res_mag.x[4], res_mag.x[5], 0, frequencies), i_shift)
#             ax2.plot(times, ephi_analytic / units.mV * units.m, "--C1", label="analytic ePhi")
#
#             ax_E.plot(times, etheta_analytic / units.mV * units.m, "--C0", label="analytic eTheta")
#             ax_E.plot(times, ephi_analytic / units.mV * units.m, "--C1", label="analytic ePhi")
#
#             ff = station.get_frequencies() / units.MHz
# #             ax2f.plot(ff[ff < 500], np.abs(station.get_frequency_spectrum()[1][ff < 500]) / units.mV * units.m, "-C0", label="4 stations lsqr eTheta")
# #             ax2f.plot(ff[ff < 500], np.abs(station.get_frequency_spectrum()[2][ff < 500]) / units.mV * units.m, "-C1", label="4 stations lsqr ePhi")
#
#             ax2f.plot(ff[ff < 500], np.abs(get_analytic_pulse_freq(res_mag.x[0], res_mag.x[1], res_mag.x[2], 0, frequencies))[ff < 500] / units.mV * units.m, "--C0")
#             ax2f.plot(ff[ff < 500], np.abs(get_analytic_pulse_freq(res_mag.x[3], res_mag.x[4], res_mag.x[5], 0, frequencies))[ff < 500] / units.mV * units.m, "--C1")
#
#             if station.has_sim_station():
#                 sim_station = station.get_sim_station()
#                 efield_sim = sim_station.get_trace()
#                 times_sim = sim_station.get_times()
#                 times_sim = (times_sim + station.get_relative_station_time()) % times_sim.max()
#                 sort_mask = np.argsort(times_sim)
#                 # ax2.plot(times_sim / units.ns, efield_sim[0] / units.mV * units.m, "--", label="simulation eR")
#                 ax2.plot(times_sim[sort_mask] / units.ns, efield_sim[1][sort_mask] / units.mV * units.m, "-C0", label="simulation eTheta")
#                 ax2.plot(times_sim[sort_mask] / units.ns, efield_sim[2][sort_mask] / units.mV * units.m, "-C1", label="simulation ePhi")
#
#                 ax_E.plot(times_sim[sort_mask][:len(times)] / units.ns, efield_sim[1][sort_mask][:len(times)] / units.mV * units.m, "-C0", label="simulation eTheta")
#                 ax_E.plot(times_sim[sort_mask][:len(times)] / units.ns, efield_sim[2][sort_mask][:len(times)] / units.mV * units.m, "-C1", label="simulation ePhi")
#                 ax_E_res.plot(times_sim[sort_mask][:len(times)] / units.ns, (etheta_analytic - efield_sim[1][sort_mask][:len(times)]) / units.mV * units.m, "-C0", label="simulation eTheta")
#                 ax_E_res.plot(times_sim[sort_mask][:len(times)] / units.ns, (ephi_analytic - efield_sim[2][sort_mask][:len(times)]) / units.mV * units.m, "-C1", label="simulation eTheta")
#
#                 ff = sim_station.get_frequencies() / units.MHz
#                 ax2f.plot(ff[ff < 500], np.abs(sim_station.get_frequency_spectrum()[1])[ff < 500] / units.mV * units.m, "-C0", label="eTheta")
#                 ax2f.plot(ff[ff < 500], np.abs(sim_station.get_frequency_spectrum()[2])[ff < 500] / units.mV * units.m, "-C1", label="ePhi")
#
#                 ax2f_phase = ax2f.twinx()
#                 ax2f_phase.plot(ff[ff < 500], np.unwrap(np.angle(sim_station.get_frequency_spectrum()[1][ff < 500])), "--C2", label="eTheta")
#                 ax2f_phase.plot(ff[ff < 500], np.unwrap(np.angle(sim_station.get_frequency_spectrum()[2][ff < 500])), "--C3", label="ePhi")
#
#             ax_E.legend(fontsize='small')
#             ax_E_res.set_xlabel("time [ns]")
#             ax_E.set_ylabel("efield [mV/m]")
#             ax_E_res.set_ylabel(r"$\Delta$")
#             ax_E.set_xlim(20, 120)
#             ax_V.set_xlim(20, 150)
#             fig_E.tight_layout()
#             fig_E.savefig("plots/event{:06d}_station{:02d}_efield.png".format(self.__counter, station.get_id()))
#             fig_V.savefig("plots/event{:06d}_station{:02d}_voltage.png".format(self.__counter, station.get_id()))
#
#             ax2.legend(fontsize="small")
#             ax2.set_xlabel("time [ns]")
#             ax2.set_ylabel("electric-field [mV/m]")
#             axf.set_xlabel("Frequency [MHz]")
#             ax2f.set_xlabel("Frequency [MHz]")
# #             ax2f.semilogy(True)
#             ax2f.set_xlim(100, 500)
#             axf.set_xlim(100, 500)
#             if station.has_sim_station():
#                 fig.suptitle("Simulation:  Zenith {:.1f}, Azimuth {:.1f}".format(np.rad2deg(zenith), np.rad2deg(azimuth)))
#             else:
#                 fig.suptitle("Data: reconstructed zenith {0:.1f}, azimuth {1:.1f}".format(np.rad2deg(zenith), np.rad2deg(azimuth)))
#             fig.tight_layout()
#             fig.subplots_adjust(top=0.95)
#
#             f1.tight_layout()
#             f1.subplots_adjust(top=0.95)
#             plt.close("all")

    def end(self):
        pass
