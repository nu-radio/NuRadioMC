from crTemplateCorrelator import crTemplateCorrelator
from crArtificialTemplateBank import crArtificialTemplateBank
import numpy as np
from tensorflow import keras

def template():

    # Load data
    data = np.load('../../data_preprocessed.npy')
    print(data[0])

    # Load generator
    generator = keras.models.load_model('time_fft_wavelet_generator_2') 

if __name__ == '__main__':
    template()