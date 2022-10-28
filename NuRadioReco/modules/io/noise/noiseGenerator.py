import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import warnings 
tf.compat.v1.experimental.output_all_intermediates(True)

class NoiseGenerator():
    '''Simple class using the generator model to generate noise'''

    def __init__(self, path_to_generator, path_to_data):
        
        # Load data
        self.dataset = np.load(path_to_data)

        # Load generator
        self.generator = keras.models.load_model(path_to_generator) 



    def generate_noise(self, number_of_events, normalized=False):
        '''Function which uses the generator to generate noise'''
        noise = np.random.randn(number_of_events, self.generator.input_shape[1])
        noise = np.expand_dims(noise, axis=-1) 
        generated_traces = self.generator.predict_on_batch(noise)
        generated_traces = generated_traces[:,:,0]
        
        if not normalized:
            generated_traces = generated_traces * self.dataset.std()+self.dataset.mean()

        
        return generated_traces

    