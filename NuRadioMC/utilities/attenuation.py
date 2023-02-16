import numpy as np
from NuRadioReco.utilities import units

import logging
logger = logging.getLogger("attenuation")
logging.basicConfig()
logger.setLevel(logging.INFO)

import scipy.interpolate
import os

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
    if(not hasattr(z, '__len__')):
        att_length = 0
    else:
        att_length = np.zeros_like(z)
    for power, coeff in enumerate(fit_values):
        att_length += coeff * z ** power

    if (not hasattr(att_length, '__len__')):
        if (att_length < min_length):
            att_length = min_length
    else:
        att_length[ att_length < min_length ] = min_length
    return att_length


def get_temperature(z):
    """
    returns the temperature in Celsius as a function of depth for South Pole

    Parameters
    ----------
    z: float
        depth in default units
    """
    # from https://icecube.wisc.edu/~araproject/radio/#icetabsorption

    z2 = np.abs(z / units.m)
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
    if(model == "SP1"):
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
        if(not hasattr(frequency, '__len__')):
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

    elif(model == "GL1"):
        att_length_75 = fit_GL1(z / units.m)
        att_length_f = att_length_75 - 0.55 * units.m * (frequency / units.MHz - 75)

        min_length = 1 * units.m
        if(not hasattr(frequency, '__len__') and not hasattr(z, '__len__')):
            if (att_length_f < min_length):
                att_length_f = min_length
        else:
            att_length_f[ att_length_f < min_length ] = min_length

    elif model == 'GL2':
        fit_values_GL2 = [1.20547286e+00, 1.58815679e-05, -2.58901767e-07, -5.16435542e-10, -2.89124473e-13, -4.58987344e-17]
        freq_slope = -0.54 * units.m / units.MHz
        freq_inter = 852.0 * units.m

        bulk_att_length_f = freq_inter + freq_slope * frequency
        att_length_f = bulk_att_length_f * np.poly1d(np.flip(fit_values_GL2))(z)

        min_length = 1 * units.m
        if (not hasattr(frequency, '__len__') and not hasattr(z, '__len__')):
            if att_length_f < min_length:
                att_length_f = min_length
        else:
            att_length_f[att_length_f < min_length] = min_length

    elif model == 'GL3':
        slopes = gl3_slope_interpolation(-z)
        offsets = gl3_offset_interpolation(-z)
        att_length_f = slopes * frequency + offsets

    elif(model == "MB1"):
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
        d = -z * 420. * units.m / d_ice;
        L = (1250.*0.08886 * np.exp(-0.048827 * (225.6746 - 86.517596 * np.log10(848.870 - (d)))))
        # this differs from the equation published in F. Wu PhD thesis UCI.
        # 262m is supposed to be the depth averaged attenuation length but the
        # integral (int(1/L, 420, 0)/420) ^ -1 = 231.21m and NOT 262m.
        att_length_f *= L / 231.21 * units.m

    else:
        raise NotImplementedError("attenuation model {} is not implemented.".format(model))

    # mask for positive z and mask for <~0 attenuation length
    if np.any(z > 0):
        logger.warning("Attenuation length is set to inf for positive z (above ice surface)")


    min_length = 1 * units.m
    if (not hasattr(frequency, '__len__') and not hasattr(z, '__len__')):
        if att_length_f < min_length:
            att_length_f = min_length
        if z > 0:
            att_length_f = np.inf
    else:
        att_length_f[att_length_f < min_length] = min_length
        att_length_f[z > 0] = np.inf
    return att_length_f


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    z = np.linspace(1*units.km,-3*units.km,1000)
    frequencies = [0.5*units.GHz] #np.linspace(0,1*units.GHz,10)
    for frequency in frequencies:
        for model in ["SP1", "GL1", "GL2", "MB1"]:
            plt.plot(-z/units.m, np.nan_to_num(get_attenuation_length(z, frequency, model)/units.m, posinf=3333), label=f"{model}")
    plt.xlabel("depth [m]")
    plt.ylabel("attenuation length [m]")
    plt.ylim(0,None)
    plt.title("attenuation length (inf masked to +3333m)")
    plt.legend()
    plt.savefig("attenuation_length_models.png")
