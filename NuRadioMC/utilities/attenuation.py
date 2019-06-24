import numpy as np
from NuRadioMC.utilities import units

model_to_int = {"SP1" : 1}

#     def get_temperature(self, z):
#         return (-51.5 + z * (-4.5319e-3 + 5.822e-6 * z))

def get_temperature(z):
    """
    returns the temperature in Celsius as a function of depth
    """
    # from https://icecube.wisc.edu/~araproject/radio/#icetabsorption

    z2 = np.abs(z / units.m)
    return 1.83415e-09 * z2**3 + (-1.59061e-08 * z2**2) + 0.00267687 * z2 + (-51.0696)

def get_attenuation_length(z, frequency, model):
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
    else:
        raise NotImplementedError("attenuation model {} is not implemented.".format(model))