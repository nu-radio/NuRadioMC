import numpy as np
import pickle
import bz2
import _pickle as cPickle
import os
from NuRadioReco.utilities import units, io_utilities

filename = 'results/ntr/dict_ntr_high_low_pb_80_180.pbz2'
print('filename', filename)
data = []
bz2 = bz2.BZ2File(filename, 'rb')
data = cPickle.load(bz2)


filename = 'results/ntr/dict_ntr_high_low_pb_80_180.pickle'
data = io_utilities.read_pickle(filename, encoding='latin1')
print(data)
print(len(data))
