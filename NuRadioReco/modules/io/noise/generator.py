import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import warnings 
import tensorflow as tf

class Generator():
    '''Discriminator (critic) model used in the WGAN'''

    def __init__(self, training = False):
        self.latent_size = 128
        self.channels = 1
        self.conv_activation = "relu"
        self.activation_function = "tanh"
        self.trace_length = 512
        self.shape = (self.trace_length, self.channels)
        self.sliding_window = 10
        self.number_of_filters = 32
        self.model = self.build_generator_transpose()
        

    
    def moving_avg(self, input):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    
        # Arguments
            input (tensor): mean and log of variance of Q(z|X)
    
        # Returns
            z (tensor): sampled latent vector
        """
        sliding_window = tf.signal.frame(
            input,
            frame_length = self.sliding_window,
            frame_step= 1,#steps
            pad_end=True,
            pad_value=0,
            axis=1,
            name='envelope_moving_average'
        )
        sliding_reshaped = keras.backend.reshape(sliding_window,(-1,self.trace_length,self.sliding_window))
        mvg_avg = keras.backend.mean(sliding_reshaped, axis=2, keepdims=True)
        return mvg_avg

    def build_generator_upsampling(self):
        """ Generator network using UpSampling1D layers """

        # Arguments used for the layers
        kwargs = {'kernel_size': 3, 'padding': 'same', 'kernel_initializer': 'he_normal'}

        noise = keras.layers.Input(shape=(self.latent_size,1), name = "noise")

        # First
        x = keras.layers.Conv1D(filters=self.latent_size//4, **kwargs)(noise)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv1D(filters=self.latent_size//4, **kwargs)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.UpSampling1D(size=2)(x)

        # Middle
        x = keras.layers.Conv1D(filters=self.latent_size//8, **kwargs)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv1D(filters=self.latent_size//8, **kwargs)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.UpSampling1D(size=2)(x)

        # Last 
        x = keras.layers.Conv1D(filters=self.latent_size//32, **kwargs)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv1D(filters=1, **kwargs)(x)


        # x = keras.layers.Reshape((self.trace_length,1))(x)
        # if self.sliding_window > 0:
        #     keras.layers.Lambda(self.moving_avg, output_shape=(self.trace_length,1), name='mvg_avg')(x)

        #     x = layers.Flatten()(x)

        generator = keras.models.Model(noise, x, name="generator")
        return generator

    def build_generator_transpose(self):
        """ Generator network using Conv1DTranspose layers """

        # Arguments used for the layers
        kwargs = {'kernel_size': 3, 'padding': 'same', 'kernel_initializer': 'he_normal'}

        # Input layer (noise)
        noise = keras.layers.Input(shape=(self.latent_size,1), name = "noise")

        
        x = keras.layers.Conv1D(filters=self.number_of_filters, **kwargs)(noise)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv1D(filters=self.number_of_filters, **kwargs)(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv1DTranspose(filters=self.number_of_filters/2, strides=2, **kwargs)(x)
        x = keras.layers.LeakyReLU()(x)

        x = keras.layers.Conv1D(filters=self.number_of_filters/2, **kwargs)(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv1D(filters=self.number_of_filters/2, **kwargs)(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv1DTranspose(filters=self.number_of_filters/4, strides=2, **kwargs)(x)
        x = keras.layers.LeakyReLU()(x)

        # x = keras.layers.Conv1D(filters=self.number_of_filters/4, **kwargs)(x)
        # x = keras.layers.LeakyReLU()(x)
        # x = keras.layers.Conv1DTranspose(filters=self.number_of_filters/8, strides=2, **kwargs)(x)
        # x = keras.layers.LeakyReLU()(x)

        x = keras.layers.Conv1D(filters=self.number_of_filters/8, **kwargs)(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv1D(filters=1, **kwargs)(x)
        


        # x = keras.layers.Reshape((self.trace_length,1))(x)
        # if self.sliding_window > 0:
        #     keras.layers.Lambda(self.moving_avg, output_shape=(self.trace_length,1), name='mvg_avg')(x)

        #     x = layers.Flatten()(x)

        generator = keras.models.Model(noise, x, name="generator")
        return generator

    
    def predict(self, noise):
        return self.model.predict(noise)

    