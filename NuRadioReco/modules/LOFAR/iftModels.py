"""
A module containing models and functions for the 
IFT electric field reconstructor for LOFAR.

Author: Karen Terveer, Phillip Frank,
        currently using parametrisation & param values
        by Simon Strähnz

"""


import nifty8.re as jft
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import dataclasses
from dataclasses import dataclass
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz

    
def parametric_amplitude(alpha, f12, b1, b2, s, freqs):
    al = (b2 - b1) / s
    return 10**alpha * f12**(b1 - b2) * freqs**b1 * (freqs**s + f12**s)**al

def phase_model(phi_0, m_phi, freqs):
    return phi_0 + (m_phi * freqs)

class EFieldModel(jft.Model):

    """
    A model to simulate an electric field signal based on various parameters
    and compute its frequency domain representation.

    Attributes:
    -----------
    alpha : jft.prior.UniformPrior
        Uniform prior for the alpha parameter.
    f12 : jft.LogNormalPrior
        Log-normal prior for the f12 parameter.
    b1 : jft.NormalPrior
        Normal prior for the b1 parameter.
    b2 : jft.NormalPrior
        Normal prior for the b2 parameter.
    s : jft.LogNormalPrior
        Log-normal prior for the s parameter.
    phi_0 : jft.NormalPrior
        Normal prior for the phi_0 parameter.
    m_phi : jft.NormalPrior
        Normal prior for the m_phi parameter.
    phi_pol : jft.NormalPrior
        Normal prior for the phi_pol parameter.
    times : numpy.ndarray
        Array of time values.
    nyq : int
        Nyquist frequency index.
    freqs : numpy.ndarray
        Array of frequency values.
    
    Methods:
    --------
    amplitude(x, neg=True, freq=None):
        Calculate the amplitude of the signal at given frequencies.
    phase(x, neg=True, freq=None):
        Calculate the phase of the signal at given frequencies.
    pol_base(x):
        Calculate the polarization base of the signal.
    trace_freq(x):
        Calculate the frequency domain trace of the signal.
    trace_time(x):
        Calculate the time domain trace of the signal.
    __call__(x):
        Call the model to compute the time domain trace.
    
    Parameters:
    -----------
    times : numpy.ndarray
        Array of time values.
    params_alpha : dict, optional
        Parameters for the alpha prior. Defaults to {"a_min": -2.5, "a_max": -2}.
    params_f12 : dict, optional
        Parameters for the f12 prior. Defaults to {"mean": 100.0, "std": 0.35}.
    params_b1 : dict, optional
        Parameters for the b1 prior. Defaults to {"mean": 0.07, "std": 0.012}.
    params_b2 : dict, optional
        Parameters for the b2 prior. Defaults to {"mean": -6.59, "std": 0.5}.
    params_s : dict, optional
        Parameters for the s prior. Defaults to {"mean": 1.0, "std": 0.4}.
    params_phi_0 : dict, optional
        Parameters for the phi_0 prior. Defaults to {"mean": -500, "std": 0.5}.
    params_m_phi : dict, optional
        Parameters for the m_phi prior. Defaults to {"mean": mt0, "std": 2}.
    params_phi_pol : dict, optional
        Parameters for the phi_pol prior. Defaults to {"mean": mt0, "std": 0.01}.
    padding : float, optional
        Padding factor for the time array. Defaults to 0.2.
    prefix : str, optional
        Prefix for parameter names. Defaults to "".
    """

    def __init__(
        self,
        times: npt.NDArray[np.float64],
        params_alpha: dict = {"a_min": -2.5, "a_max": -2},
        params_f12: dict = {"mean": np.log(100.0), "std": 0.35},
        params_b1: dict = {"mean": 0.07, "std": 0.012},
        params_b2: dict = {"mean": -6.59, "std": 0.5},
        params_s: dict = {"mean": 1.0, "std": 0.4}, 
        params_phi_0: dict = {"mean": -500, "std": 0.5},
        params_m_phi: dict = {"mean": -1048.566181, "std": 20}, 
        params_phi_pol: dict = {"mean": 2.5146966741583343, "std": 0.8},
        padding=0.2,
        prefix="",
    ):
        self.alpha = jft.prior.UniformPrior(
            **params_alpha, shape=(1,), name=prefix + "alpha"
        )
        self.f12 = jft.LogNormalPrior(**params_f12, shape=(1,), name=prefix + "f12")
        self.b1 = jft.NormalPrior(**params_b1, shape=(1,), name=prefix + "b1")
        self.b2 = jft.NormalPrior(**params_b2, shape=(1,), name=prefix + "b2")
        self.s = jft.LogNormalPrior(**params_s, shape=(1,), name=prefix + "s")
        self.phi_0 = jft.NormalPrior(**params_phi_0, shape=(1,), name=prefix + "phi_0")
        self.m_phi = jft.NormalPrior(**params_m_phi, shape=(1,), name=prefix + "m_phi")
        self.phi_pol = jft.NormalPrior(
            **params_phi_pol, shape=(), name=prefix + "phi_pol"
        )

        self.ops = (self.alpha, self.f12, self.b1, self.b2, self.s)
        self.ops2 = (self.phi_0, self.m_phi)
        self.times = times
        resol = times[1] - times[0]
        assert np.allclose(times[1:] - times[:-1], resol * np.ones(times.size - 1))
        npix = int((1.0 + padding) * times.size)
        fmax = 0.5 / resol * 1000 #WARNING converting GHz to MHz
        self.nyq = (npix - 1) // 2
        self.freqs = np.roll(np.arange(npix) - self.nyq, -self.nyq) * fmax / (npix // 2)

        init = (
            self.alpha.init
            | self.f12.init
            | self.b1.init
            | self.b2.init
            | self.phi_0.init
            | self.m_phi.init
            | self.s.init
            | self.phi_pol.init
        )
        super().__init__(init=init)

    def amplitude(self, x, neg=True, freq=None):

        """
        Calculate the amplitude of the signal at given frequencies.

        Parameters:
        -----------
        x : array-like
            Input array for parameter values.
        neg : bool, optional
            If True, use the negative frequencies. Defaults to True.
        freq : array-like, optional
            Frequencies to use. If None, defaults to self.freqs.

        Returns:
        --------
        amplitude : array-like
            Calculated amplitude values.
        """

        if freq is None:
            fr = self.freqs if neg else self.freqs[self.nyq + 1]
        else:
            fr = freq
        return parametric_amplitude(*(oo(x) for oo in self.ops), np.abs(fr))

    def phase(self, x, neg=True, freq=None):

        """
        Calculate the phase of the signal at given frequencies.

        Parameters:
        -----------
        x : array-like
            Input array for parameter values.
        neg : bool, optional
            If True, use the negative frequencies. Defaults to True.
        freq : array-like, optional
            Frequencies to use. If None, defaults to self.freqs.

        Returns:
        --------
        phase : array-like
            Calculated phase values.
        """

        if freq is None:
            fr = self.freqs if neg else self.freqs[self.nyq + 1]
        else:
            fr = freq
        return jnp.exp(1j * phase_model(*(oo2(x) for oo2 in self.ops2), fr))

    def pol_base(self, x):

        """
        Calculate the polarization base of the signal.

        Parameters:
        -----------
        x : array-like
            Input array for parameter values.

        Returns:
        --------
        pol_base : array-like
            Polarization base values.
        """

        phi = self.phi_pol(x)
        return jnp.array([jnp.cos(phi), jnp.sin(phi)])

    def trace_freq(self, x):

        """
        Calculate the frequency domain trace of the signal.

        Parameters:
        -----------
        x : array-like
            Input array for parameter values.

        Returns:
        --------
        trace_freq : array-like
            Frequency domain trace values.
        """

        res = (self.amplitude(x) * self.phase(x))[:, jnp.newaxis]
        res *= self.pol_base(x)[jnp.newaxis, :]
        return res

    def trace_time(self, x):

        """
        Calculate the time domain trace of the signal.

        Parameters:
        -----------
        x : array-like
            Input array for parameter values.

        Returns:
        --------
        trace_time : array-like
            Time domain trace values.
        """
        # TODO volume factor?!
        return jnp.fft.ifft(self.trace_freq(x), axis=0).real[: self.times.size]

    def __call__(self, x):

        """
        Call the model to compute the time domain trace.

        Parameters:
        -----------
        x : array-like
            Input array for parameter values.

        Returns:
        --------
        trace_time : array-like
            Time domain trace values.
        """

        return self.trace_time(x)


class AntennaResponse(jft.Model):

    """
    A model representing the antenna response to an electric field signal.

    Attributes:
    -----------
    eFieldModel : EFieldModel
        The electric field model representing the input signal.
    response : numpy.ndarray
        The response of the antenna as a function of frequency.
    
    Methods:
    --------
    __init__(eFieldModel, response=None):
        Initialize the AntennaResponse model with an electric field model and optionally a response array.
    
    trace_freq(x):
        Compute the frequency domain trace of the signal after applying the antenna response.
    
    trace_time(x):
        Compute the time domain trace of the signal after applying the antenna response.
    
    __call__(x):
        Call the model to compute the time domain trace of the signal.
    
    Parameters:
    -----------
    eFieldModel : EFieldModel
        The electric field model representing the input signal.
    response : numpy.ndarray, optional
        The response of the antenna as a function of frequency. If None, it is computed using the `antenna_res` function.
    """
    

    eFieldModel: EFieldModel = dataclasses.field(metadata=dict(static=False))
    response: npt.NDArray[np.float64] = dataclasses.field(metadata=dict(static=False))

    def __init__(self, eFieldModel: EFieldModel, response: npt.NDArray = None):

        """
        Initialize the AntennaResponse model.

        Parameters:
        -----------
        eFieldModel : EFieldModel
            The electric field model representing the input signal.
        response : numpy.ndarray, optional
            The response of the antenna as a function of frequency. 
        """

        self.eFieldModel = eFieldModel
        if response is None:
            raise ValueError("Response must be provided")
        self.response = response
        super().__init__(init=self.eFieldModel.init)

    def trace_freq(self, x):
    
        """
        Compute the frequency domain trace of the signal after applying the antenna response.

        Parameters:
        -----------
        x : array-like
            Input array for parameter values.

        Returns:
        --------
        trace_freq : array-like
            Frequency domain trace values after applying the antenna response.
        """

        return (self.response @ self.eFieldModel.trace_freq(x)[..., jnp.newaxis])[
            ..., 0
        ]

    def trace_time(self, x):

        """
        Compute the time domain trace of the signal after applying the antenna response.

        Parameters:
        -----------
        x : array-like
            Input array for parameter values.

        Returns:
        --------
        trace_time : array-like
            Time domain trace values after applying the antenna response.
        """

        # TODO volume factor
        
        return jnp.fft.ifft(self.trace_freq(x), axis=0).real[
            : self.eFieldModel.times.size
        ]

        

    def __call__(self, x):

        """
        Call the model to compute the time domain trace of the signal.

        Parameters:
        -----------
        x : array-like
            Input array for parameter values.

        Returns:
        --------
        trace_time : array-like
            Time domain trace values after applying the antenna response.
        """
        
        return self.trace_time(x)

def process_trace(trace, times, start_time, end_time, cut_fraq=0.1):

    """
    Process a signal trace by trimming edges, selecting a specific time window,
    and organizing pre- and post-window traces.

    Parameters:
    -----------
    trace : array-like
        The signal trace data.
    times : array-like
        The corresponding time values for the trace data.
    start_time : float
        The starting time for the selected window.
    end_time : float
        The ending time for the selected window.
    cut_fraq : float, optional
        The fraction of the trace to cut from the beginning and the end for edge trimming. 
        Defaults to 0.1.

    Returns:
    --------
    trace : array-like
        The processed trace data within the specified time window.
    concatenated_traces : array-like
        The concatenated traces from before and after the selected time window.

    Notes:
    ------
    This function performs the following steps:
    1. Trims the edges of the trace based on the `cut_fraq` parameter.
    2. Selects a specific window of the trace between `start_time` and `end_time`.
    3. Extracts and reshapes traces from before and after the selected window.
    4. Returns the processed trace and the concatenated pre- and post-window traces.
    """

    N = times.size
    cut = int(N * cut_fraq)
    trace, times = trace[cut:-cut], times[cut:-cut]
    startid = np.where(times >= start_time)[0][0]
    endid = np.where(times <= end_time)[0][-1]
    Nids = endid - startid
    trace_pre = trace[:startid]
    trace_post = trace_pre[endid:]
    trace = trace[startid:endid]

    trace_pre = trace_pre[:Nids * (trace_pre.size // Nids)].reshape((-1, Nids))
    trace_post = trace_post[:Nids * (trace_post.size // Nids)].reshape((-1, Nids))

    return trace, np.concatenate((trace_pre, trace_post), axis=0)

def stationary_noise(trace, time_cut = 0.25, variance_cut = 1E-4, debug=False):

    """
    Process a signal trace to compute the stationary noise characteristics and 
    return a Noise dataclass encapsulating the noise properties.

    Parameters:
    -----------
    trace : array-like
        The signal trace data.
    time_cut : float, optional
        The fraction of the trace to retain for computing the noise characteristics.
        Defaults to 0.25.
    variance_cut : float, optional
        The minimum allowable variance for the noise eigenvalues. 
        Eigenvalues below this threshold are set to this value. 
        Defaults to 1E-4.
    debug : bool, optional

    Returns:
    --------
    Noise : Noise
        A dataclass instance encapsulating the noise covariance matrix, its eigenvalues, 
        and eigenvectors, along with the standard deviation of the input trace.

    Notes:
    ------
    This function performs the following steps:
    1. Normalizes the trace by its standard deviation.
    2. Reshapes the trace and computes the time-domain covariance.
    3. Trims the covariance function based on the `time_cut` parameter.
    4. Constructs the noise covariance matrix using the trimmed covariance function.
    5. Performs an eigen decomposition on the covariance matrix.
    6. Adjusts the eigenvalues based on the `variance_cut` parameter.
    7. Returns a Noise dataclass encapsulating the noise properties.
    """
    
    std = np.sqrt(np.mean(trace**2))

    trace /= std
    trace = np.moveaxis(trace, -1, 1)
    trace = trace.reshape((-1,) + trace.shape[2:])
    T = (trace[..., np.newaxis] * trace[..., np.newaxis, :]).mean(axis=0).ravel()
    ids = np.arange(trace.shape[-1], dtype=int)
    ids = np.abs(ids[:, np.newaxis] - ids[np.newaxis, :])

    t = np.zeros(trace.shape[-1])
    wgts = np.zeros_like(t)
    ids = ids.ravel()
    np.add.at(t, ids, T)
    np.add.at(wgts, ids, 1)
    t /= wgts

    if debug==True:
        plt.plot(t)
        t[int(t.size*time_cut):] = 0.
        plt.plot(t)
        plt.xlabel('Time')
        plt.ylabel('Covariance')
        plt.show()

    N = toeplitz(t)
    v, U = np.linalg.eigh(N)
    print(np.min(v), np.max(v))
    v[v < variance_cut] = variance_cut
    print(np.min(v), np.max(v))

    @dataclass
    class Noise:

        """
        A dataclass encapsulating the noise properties of a signal trace.

        Attributes:
        -----------
        v : numpy.ndarray
            The eigenvalues of the noise covariance matrix.
        U : numpy.ndarray
            The eigenvectors of the noise covariance matrix.
        sig : float
            The standard deviation of the input trace.

        Methods:
        --------
        _cov(inverse=False):
            Compute the noise covariance matrix or its inverse.
        
        _amp(inverse=False):
            Compute the noise amplitude matrix or its inverse.
        
        cov(inverse=False):
            Compute the scaled noise covariance matrix or its inverse.
        
        amp(inverse=False):
            Compute the scaled noise amplitude matrix or its inverse.
        
        noise_cov_inv(x):
            Apply the inverse noise covariance matrix to a vector.
        
        noise_std_inv(x):
            Apply the inverse noise standard deviation matrix to a vector.

        Parameters:
        -----------
        v : numpy.ndarray
            The eigenvalues of the noise covariance matrix.
        U : numpy.ndarray
            The eigenvectors of the noise covariance matrix.
        sig : float
            The standard deviation of the input trace.
        """
        
        v: npt.NDArray[np.float64]
        U: npt.NDArray[np.float64]
        sig: float
        def __init__(self, v, U, sig):
            self.v = v
            self.U = U
            self.sig = sig

        def _cov(self, inverse=False):
            v = 1./self.v if inverse else self.v
            return (U @ np.diag(v) @ U.T)

        def _amp(self, inverse=False):
            if inverse:
                return np.diag(1. / np.sqrt(v)) @ U.T
            return U @ np.diag(np.sqrt(v))

        def cov(self, inverse=False):
            scale = self.sig**-2 if inverse else self.sig**2
            return self._cov(inverse=inverse) * scale

        def amp(self, inverse=False):
            scale = self.sig**-1 if inverse else self.sig
            return scale * self._amp(inverse=inverse)

        def noise_cov_inv(self, x):
            return (self._cov(inverse=True) @ x) * self.sig**-2

        def noise_std_inv(self, x):
            return (self._amp(inverse=True) @ x) * self.sig**-1
        
    return Noise(v, U, std)
