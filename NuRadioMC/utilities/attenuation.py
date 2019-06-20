import numpy as np
from NuRadioMC.utilities import units

model_to_int = {"SP1" : 1, "GL1" : 2}
cache_ice = {}

def cache_GL1():
    # Model for Greenland. Taken from DOI: https://doi.org/10.3189/2015JoG15J057
    att_length = [453.19, 461.92, 472.39, 483.73, 492.46, 502.05, 510.78, 519.50, 528.23,
    534.33, 543.06, 550.91, 561.38, 575.34, 584.06, 595.40, 605.87, 617.21, 627.68, 639.90,
    651.24, 660.83, 673.05, 683.52, 693.98, 702.71, 714.05, 728.88, 741.97, 760.29, 775.99,
    786.46, 799.54, 810.88, 823.10, 836.18, 848.40, 860.61, 871.08, 883.29, 899.87, 911.21,
    926.04, 938.25, 948.72, 960.94, 974.02, 986.24, 997.58, 1008.05, 1022.01, 1031.61,
    1043.82, 1056.91, 1065.63, 1076.98, 1088.32, 1099.66, 1113.63, 1121.48, 1133.7, 1141.55,
    1150.28, 1159.89, 1168.62, 1174.74, 1179.11, 1180.87, 1180.01, 1179.15, 1174.80, 1169.59,
    1165.25, 1160.03, 1153.94, 1149.59, 1144.37, 1140.03, 1137.42, 1134.82, 1133.97, 1135.73,
    1136.62, 1140.12, 1143.63, 1148.00, 1154.12, 1153.27, 1153.28, 1152.42]
    att_length = np.array(att_length) * units.m

    depth = [-3038.15, -3007.84, -2988.93, -2962.42, -2947.29, -2935.95, -2913.23, -2901.89, -2890.55,
    -2875.41, -2860.27, -2845.14, -2826.22, -2811.10, -2792.18, -2773.26, -2754.34, -2739.22, -2716.51,
    -2701.39, -2682.47, -2663.55, -2644.64, -2633.30, -2614.38, -2603.04, -2580.34, -2565.23, -2538.73,
    -2512.26, -2493.36, -2470.65, -2451.74, -2440.41, -2417.71, -2395.01, -2379.89, -2357.19, -2342.06,
    -2319.35, -2296.67, -2285.34, -2262.64, -2243.73, -2221.02, -2202.11, -2179.41, -2156.70, -2126.41,
    -2118.87, -2092.38, -2069.66, -2043.16, -2024.26, -1997.74, -1967.45, -1944.74, -1918.24, -1880.37,
    -1853.86, -1827.36, -1789.47, -1755.37, -1702.31, -1641.67, -1588.60, -1524.14, -1459.67, -1395.19,
    -1330.71, -1285.18, -1194.14, -1129.64, -1061.35, -1000.64, -947.53, -883.03, -810.95, -746.46,
    -678.19, -606.12, -537.86, -458.22, -378.59, -306.54, -238.29, -173.84, -97.98, -44.89, -3.16]
    depth = np.array(depth) * units.m

    att_length_75_interp = interp1d(depth, att_length,
    bounds_error = False, fill_value=(att_length[0],att_length[-1])) # Greenland attenuation length for 75 MHz

    return att_length_75_interp(z)

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
    elif(model == "GL1"):

        if("GL1" not in cache_ice):
            cache_ice["GL1"] = cache_GL1()

        att_length_75 = cache_ice["GL1"]
        att_length_f = att_length_75 - 0.55 * units.m * (frequency/units.MHz - 75)
        return att_length_f
    else:
        raise NotImplementedError("attenuation model {} is not implemented.".format(model))
