from NuRadioReco.utilities import units

import scipy.interpolate
import functools
import numpy as np
import os
import re

import logging
logger = logging.getLogger("NuRadioMC.attenuation")



model_to_int = {"SP1": 1, "GL1": 2, "MB1": 3, "GL2": 4, "GL3": 5}

gl3_parameters = np.genfromtxt(
    os.path.join(os.path.dirname(__file__), 'data/GL3_params.csv'),
    delimiter=','
)

gl3_slope_interpolation = scipy.interpolate.interp1d(
    gl3_parameters[:, 0],
    gl3_parameters[:, 1],
    bounds_error=False,
    fill_value=(gl3_parameters[0, 1], gl3_parameters[-1, 1])
)

gl3_offset_interpolation = scipy.interpolate.interp1d(
    gl3_parameters[:, 0],
    gl3_parameters[:, 2],
    bounds_error=False,
    fill_value=(gl3_parameters[0, 2], gl3_parameters[-1, 2])
)

def read_grip_temperature():
    """
    Read temperature file from grip.

    Data file from footnote 5 from https://arxiv.org/pdf/2201.07846

    Returns
    -------
    depths: array of floats
        Depths in meters (positive!).
    temps: array of floats
        Temperatures in Kelvin.
    """
    path = os.path.join(os.path.dirname(__file__), 'data/griptemp.txt')

    depths = []
    temps = []

    with open(path) as f:
        # search for "DATA:" section in file
        search = re.search("DATA:", f.read())
        assert search is not None, "No DATA section found in file"

        # Jumpe in file to "DATA:" section
        f.seek(search.span()[0])

        for line in f.read().split("\n"):
            if re.match(r'^\d', line) is not None:  # match lines beginning with a digit
                line = line.strip("\n")  # remove newline character
                depth, temp = [float(d) for d in line.split("\t")]
                depths.append(depth)
                temps.append(temp)

    return np.asarray(depths, dtype=float), np.asarray(temps, dtype=float) + 273.15  # convert to Kelvin

@functools.lru_cache(maxsize=1)
def grip_temperature_profile():
    """
    Returns interpolation function for temperature profile from Greenland Ice Core Project (GRIP) borehole measurements.

    Returns
    -------
    func : scipy.interpolate.interp1d
        f(depth) -> temperature in Kelvin. Depth is positivly defined: 0 is the surface, 3000 is the bottom.
    """
    d, t = read_grip_temperature()
    return scipy.interpolate.interp1d(d, t, fill_value="extrapolate")

def get_grip_temperature(depth):
    """
    Returns temperature at a given depth from Greenland Ice Core Project (GRIP) borehole measurements.

    Parameters
    ----------
    depth : float
        Depth in meters. Positive values are below the surface.

    Returns
    -------
    float
        Temperature in Kelvin.
    """
    return grip_temperature_profile()(depth)

def fit_GL1(z):
    """
    Returns the attenuation length at 75 MHz as a function of depth for Greenland
    # Model for Greenland. Taken from DOI: https://doi.org/10.3189/2015JoG15J057

    Parameters
    ----------
    z: float
        depth in default units
    """

    fit_values = [1.16052586e+03, 6.87257150e-02, -9.82378264e-05,
        -3.50628312e-07, -2.21040482e-10, -3.63912864e-14]
    min_length = 100 * units.m

    if not hasattr(z, '__len__'):
        att_length = 0
    else:
        att_length = np.zeros_like(z)

    for power, coeff in enumerate(fit_values):
        att_length += coeff * z ** power

    if not hasattr(att_length, '__len__'):
        if (att_length < min_length):
            att_length = min_length
    else:
        att_length[att_length < min_length] = min_length

    return att_length

def get_temperature(z):
    """
    Returns the temperature in Celsius as a function of depth for South Pole

    Parameters
    ----------
    z: float
        depth in default units
    """
    # from https://icecube.wisc.edu/~araproject/radio/#icetabsorption

    z2 = np.abs(z) / units.m
    return 1.83415e-09 * z2 ** 3 + (-1.59061e-08 * z2 ** 2) + 0.00267687 * z2 + (-51.0696)


def get_attenuation_length(z, frequency, model):
    """
    Get attenuation length in ice for different ice models

    Parameters
    ----------
    z: float
        depth in default units
    frequency: float
        frequency of signal in default units
    model: string
        Ice model for attenuation length. Options:

        * SP1: South Pole model, see various compilation
        * GL1: Greenland model, see https://arxiv.org/abs/1409.5413
        * GL2: 2021 Greenland model, using the Bogorodsky model for depth dependence
          see: https://arxiv.org/abs/2201.07846, specifically Fig. 7
        * GL3: 2021 Greenland model, using the MacGregor model for depth dependence
          see: https://arxiv.org/abs/2201.07846, specifically Fig. 7
        * MB1: Moore's Bay Model, from 10.3189/2015JoG14J214 and
          Phd Thesis C. Persichilli (depth dependence)

    """
    if model == "SP1":
        t = get_temperature(z)
        f0 = 0.0001
        f2 = 3.16
        w0 = np.log(f0)
        w1 = 0.0
        w2 = np.log(f2)
        w = np.log(frequency / units.GHz)
        b0 = -6.74890 + t * (0.026709 - t * 0.000884)
        b1 = -6.22121 - t * (0.070927 + t * 0.001773)
        b2 = -4.09468 - t * (0.002213 + t * 0.000332)
        if not hasattr(frequency, '__len__'):
            if (frequency < 1. * units.GHz):
                a = (b1 * w0 - b0 * w1) / (w0 - w1)
                bb = (b1 - b0) / (w1 - w0)
            else:
                a = (b2 * w1 - b1 * w2) / (w1 - w2)
                bb = (b2 - b1) / (w2 - w1)
        else:
            a = np.ones_like(frequency) * (b2 * w1 - b1 * w2) / (w1 - w2)
            bb = np.ones_like(frequency) * (b2 - b1) / (w2 - w1)
            a[frequency < 1. * units.GHz] = (b1 * w0 - b0 * w1) / (w0 - w1)
            bb[frequency < 1. * units.GHz] = (b1 - b0) / (w1 - w0)

        att_length_f = 1. / np.exp(a + bb * w)

    elif model == "GL1":
        att_length_75 = fit_GL1(z / units.m)
        att_length_f = att_length_75 - 0.55 * units.m * (frequency / units.MHz - 75)

    elif model == 'GL2':
        fit_values_GL2 = [1.20547286e+00, 1.58815679e-05, -2.58901767e-07, -5.16435542e-10, -2.89124473e-13, -4.58987344e-17]
        freq_slope = -0.54 * units.m / units.MHz
        freq_inter = 852.0 * units.m

        bulk_att_length_f = freq_inter + freq_slope * frequency
        att_length_f = bulk_att_length_f * np.poly1d(np.flip(fit_values_GL2))(z)

    elif model == 'GL3':
        slopes = gl3_slope_interpolation(-z)
        offsets = gl3_offset_interpolation(-z)
        # # restric frequency to prevent negative attenuation lengths
        # if hasattr(frequency, '__len__'):
        #     frequency[frequency > 0.6 * units.GHz] = 0.6 * units.GHz
        # else:
        #     if frequency > 0.6 * units.GHz:
        #         frequency = 0.6 * units.GHz

        att_length_f = slopes * frequency + offsets

    elif model == "MB1":
        # 10.3189/2015JoG14J214 measured the depth-averaged attenuation length as a function of frequency
        # the derived parameterization assumed a reflection coefficient of 1
        # however a reflection coefficient of 0.82 was measured which results in a slight increase of the derived
        # attenution length (correction factor below)
        R = 0.82
        d_ice = 576 * units.m
        att_length_f = 460 * units.m - 180 * units.m / units.GHz * frequency
        att_length_f *= (1 + att_length_f / (2 * d_ice) * np.log(R)) ** -1  # additional correction for reflection coefficient being less than 1.

        # The temperature dependence of the attenuation length is independent of the frequency dependence
        # the relationship between temparature and L is from 10.1063/1.363582
        # the temparature profile of the Ross Ice shelf is from 10.1126/science.203.4379.433 and rescaled to the shelf
        # thickness of the ARIANNA site (almost linear from -28 - -2deg C.
        # this theoretical depth dependence is scaled to match the depth-averaged measurement of the attenuation length.
        d = -z * 420. * units.m / d_ice
        L = (1250.*0.08886 * np.exp(-0.048827 * (225.6746 - 86.517596 * np.log10(848.870 - (d)))))
        # this differs from the equation published in F. Wu PhD thesis UCI.
        # 262m is supposed to be the depth averaged attenuation length but the
        # integral (int(1/L, 420, 0)/420) ^ -1 = 231.21m and NOT 262m.
        att_length_f *= L / 231.21 * units.m

    else:
        raise NotImplementedError("attenuation model {} is not implemented.".format(model))

    if hasattr(z, '__len__') and not np.any(z <= 0):
        logger.warning("You requested the attenuation length for exlusively positive depths, i.e., for air. Return inf for all frequencies.")

    min_length = 1 * units.m
    if not hasattr(frequency, '__len__') and not hasattr(z, '__len__'):
        if att_length_f < min_length:
            att_length_f = min_length
        if z > 0:
            att_length_f = np.inf
    else:
        att_length_f[att_length_f < min_length] = min_length
        att_length_f[z > 0] = np.inf

    return att_length_f


try:
    from numba import jit

    def gl3_slope_interpolation_f(x):
        """
        Interpolation function for the slope of the GL3 model.

        Parameters
        ----------
        x : array-like
            Depth in meters. Positive values are below the surface.

        Returns
        -------
        array-like
            Slope of the attenuation length as a function of depth.
        """
        if x < np.amin(gl3_parameters[:, 0]):
            return gl3_parameters[0, 1]
        if x > np.amax(gl3_parameters[:, 0]):
            return gl3_parameters[-1, 1]

        return np.interp(x, gl3_parameters[:, 0], gl3_parameters[:, 1])


    def gl3_offset_interpolation_f(x):
        """
        Interpolation function for the offset of the GL3 model.

        Parameters
        ----------
        x : array-like
            Depth in meters. Positive values are below the surface.

        Returns
        -------
        array-like
            Offset of the attenuation length as a function of depth.
        """
        if x < np.amin(gl3_parameters[:, 0]):
            return np.ones_like(x) * gl3_parameters[0, 2]
        if x > np.amax(gl3_parameters[:, 0]):
            return np.ones_like(x) * gl3_parameters[-1, 2]
        return np.interp(x, gl3_parameters[:, 0], gl3_parameters[:, 2])


    gl3_slope_interpolation = jit(gl3_slope_interpolation_f, nopython=True, cache=True)
    gl3_offset_interpolation = jit(gl3_offset_interpolation_f, nopython=True, cache=True)
    get_temperature = jit(get_temperature, nopython=True, cache=True)

except ImportError:
    pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from NuRadioMC.SignalProp.CPPAnalyticRayTracing import wrapper as cpp_wrapper

    z = np.linspace(-10 * units.m, -3 * units.km, 1000)

    fig, ax = plt.subplots()
    for idx, freq in enumerate([0.55, 0.6, 0.7, 0.8, 0.9, 1]):
        for model in ["GL1", "GL3"]:
            fmt = f"C{idx}" + "-" if model == "GL1" else f"C{idx}" + "--"
            atteniaton_length = np.array([get_attenuation_length(d, freq * units.GHz, model) for d in z])
            label = ""
            if model == "GL1":
                label = f"{freq} GHz"
            ax.plot(atteniaton_length, z, fmt, lw=1, label=label)

    ax.plot(np.nan, np.nan, "k-", lw=1, label="GL1")
    ax.plot(np.nan, np.nan, "k--", lw=1, label="GL3")

    ax.set_xlabel("attenuation length [m]")
    ax.set_ylabel("depth [m]")
    ax.legend()
    fig.tight_layout()
    plt.show()
