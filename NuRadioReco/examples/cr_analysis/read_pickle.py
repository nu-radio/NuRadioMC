import numpy as np
import pickle
import os
from NuRadioReco.utilities import units, io_utilities

#filename = 'output_threshold_estimate/estimate_threshold_pb_80_180_i50000.pickle'
#filename = 'output_threshold_final/final_threshold_pb_80_180_i2000_1.pickle'
filename = 'results/dict_ntr_pb_80_180.pickle'

print('filename', filename)
data = []
data = io_utilities.read_pickle(filename, encoding='latin1')
print(data)
print(len(data))

