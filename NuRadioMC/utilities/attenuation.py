import numpy as np
from NuRadioReco.utilities import units

model_to_int = {"SP1" : 1, "GL1" : 2, "MB1" : 3}


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

        return 1. / np.exp(a + bb * w)
    elif(model == "GL1"):

        att_length_75 = fit_GL1(z / units.m)
        att_length_f = att_length_75 - 0.55 * units.m * (frequency / units.MHz - 75)

        min_length = 100 * units.m
        if(not hasattr(frequency, '__len__') and not hasattr(z, '__len__')):
            if (att_length_f < min_length):
                att_length_f = min_length
        else:
            att_length_f[ att_length_f < min_length ] = min_length

        return att_length_f
    elif(model == "MB1"):
        # 10.3189/2015JoG14J214 measured the depth-averaged attenuation length as a function of frequency
        # the derived parameterization assumed a reflection coefficient of 1
        # however a reflection coefficient of 0.82 was measured which results in a slight increase of the derived
        # attenution length (correction factor below)
        R = 0.82
        d_ice = 576 * units.m
        att_length = 460 * units.m - 180 * units.m / units.GHz * frequency
        att_length *= (1 + att_length / (2 * d_ice) * np.log(R)) ** -1  # additional correction for reflection coefficient being less than 1.

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
        att_length *= L / 231.21 * units.m

        return att_length
    else:
        raise NotImplementedError("attenuation model {} is not implemented.".format(model))
