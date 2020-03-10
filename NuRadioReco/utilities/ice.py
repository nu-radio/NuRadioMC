import numpy as np

"""
implementation of ice models
"""


def get_refractive_index(depth, site='southpole'):
    if(depth <= 0):
        return 1.3
    else:
        return 1.000293
