import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import warnings 
import tensorflow as tf
tf.compat.v1.experimental.output_all_intermediates(True)

class Generator():
    '''Generator model used in the WGAN'''

    def __init__(self, trace_length, latent_size):

        # Settings for latent size and trace length
        self.latent_size = latent_size
        self.trace_length = trace_length

        # Layer settings
        self.number_of_filters = 32

        # Decide model architecture
        self.model = self.build_lstm_conv()
        

    def build_generator_upsampling(self):

        """ Generator network using UpSampling1D layers """

        # Arguments used for the layers
        kwargs = {'kernel_size': 3, 'padding': 'same', 'kernel_initializer': 'he_normal'}

        noise = keras.layers.Input(shape=(self.latent_size,1), name = "noise")

        x = keras.layers.Conv1D(filters=self.number_of_filters//2, **kwargs)(noise)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv1D(filters=self.number_of_filters//2, **kwargs)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.UpSampling1D(size=2)(x)

        x = keras.layers.Conv1D(filters=self.number_of_filters//4, **kwargs)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv1D(filters=self.number_of_filters/4, **kwargs)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.UpSampling1D(size=2)(x)

        if self.trace_length == 512:

            x = keras.layers.Conv1D(filters=self.number_of_filters//8, **kwargs)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.Conv1D(filters=1, **kwargs)(x)

        if self.trace_length == 2048:
            x = keras.layers.Conv1D(filters=self.number_of_filters//8, **kwargs)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.Conv1D(filters=self.number_of_filters//8, **kwargs)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.UpSampling1D(size=2)(x)

            x = keras.layers.Conv1D(filters=self.number_of_filters//8, **kwargs)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.Conv1D(filters=self.number_of_filters//8, **kwargs)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.UpSampling1D(size=2)(x)

            x = keras.layers.Conv1D(filters=self.number_of_filters//16, **kwargs)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.Conv1D(filters=1, **kwargs)(x)


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

        if self.trace_length == 512:

            x = keras.layers.Conv1D(filters=self.number_of_filters/8, **kwargs)(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.Conv1D(filters=1, **kwargs)(x)


        if self.trace_length == 2048:

            x = keras.layers.Conv1D(filters=self.number_of_filters/4, **kwargs)(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.Conv1D(filters=self.number_of_filters/4, **kwargs)(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.Conv1DTranspose(filters=self.number_of_filters/8, strides=2, **kwargs)(x)
            x = keras.layers.LeakyReLU()(x)

            x = keras.layers.Conv1D(filters=self.number_of_filters/8, **kwargs)(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.Conv1D(filters=self.number_of_filters/8, **kwargs)(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.Conv1DTranspose(filters=self.number_of_filters/16, strides=2, **kwargs)(x)
            x = keras.layers.LeakyReLU()(x)

            x = keras.layers.Conv1D(filters=self.number_of_filters/16, **kwargs)(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.Conv1D(filters=1, **kwargs)(x)

       
        generator = keras.models.Model(noise, x, name="generator")
        return generator

    def build_lstm_conv(self):

        '''Model using LSTM and convolutional layers'''

        # Arguments used for the convolutional layers
        kwargs = {'kernel_size': 3, 'padding': 'same', 'kernel_initializer': 'he_normal'}

        generator = keras.models.Sequential()
        
        # Input layer (noise)
        generator.add(keras.layers.Input(shape=(self.latent_size,1), name = "noise"))

        # LSTM
        generator.add(keras.layers.Bidirectional(keras.layers.LSTM(16, return_sequences=True)))

        # Convolutional layers
        generator.add(keras.layers.Conv1D(filters=self.number_of_filters, **kwargs))
        generator.add(keras.layers.LeakyReLU())
        generator.add(keras.layers.Conv1D(filters=self.number_of_filters, **kwargs))
        generator.add(keras.layers.LeakyReLU())
        generator.add(keras.layers.Conv1DTranspose(filters=self.number_of_filters/2, strides=2, **kwargs))
        generator.add(keras.layers.LeakyReLU())

        generator.add(keras.layers.Conv1D(filters=self.number_of_filters/2, **kwargs))
        generator.add(keras.layers.LeakyReLU())
        generator.add(keras.layers.Conv1D(filters=self.number_of_filters/2, **kwargs))
        generator.add(keras.layers.LeakyReLU())
        generator.add(keras.layers.Conv1DTranspose(filters=self.number_of_filters/4, strides=2, **kwargs))
        generator.add(keras.layers.LeakyReLU())

        if self.trace_length == 512:
            generator.add(keras.layers.Conv1D(filters=self.number_of_filters/8, **kwargs))
            generator.add(keras.layers.LeakyReLU())
            generator.add(keras.layers.Conv1D(1, **kwargs))

        if self.trace_length == 2048:

            generator.add(keras.layers.Conv1D(filters=self.number_of_filters/4, **kwargs))
            generator.add(keras.layers.LeakyReLU())
            generator.add(keras.layers.Conv1D(filters=self.number_of_filters/4, **kwargs))
            generator.add(keras.layers.LeakyReLU())
            generator.add(keras.layers.Conv1DTranspose(filters=self.number_of_filters/8, strides=2, **kwargs))
            generator.add(keras.layers.LeakyReLU())

            generator.add(keras.layers.Conv1D(filters=self.number_of_filters/8, **kwargs))
            generator.add(keras.layers.LeakyReLU())
            generator.add(keras.layers.Conv1D(filters=self.number_of_filters/8, **kwargs))
            generator.add(keras.layers.LeakyReLU())
            generator.add(keras.layers.Conv1DTranspose(filters=self.number_of_filters/16, strides=2, **kwargs))
            generator.add(keras.layers.LeakyReLU())

            generator.add(keras.layers.Conv1D(filters=self.number_of_filters/16, **kwargs))
            generator.add(keras.layers.LeakyReLU())
            generator.add(keras.layers.Conv1D(1, **kwargs))

        return generator

    def build_lstm(self):

        ''' Model using only LSTM layers '''

        generator = keras.models.Sequential()
        
        # Input layer (noise)
        generator.add(keras.layers.Input(shape=(self.latent_size,1), name = "noise"))

        # LSTM layers
        generator.add(keras.layers.Bidirectional(keras.layers.LSTM(25, return_sequences=True)))
        generator.add(keras.layers.Bidirectional(keras.layers.LSTM(25, return_sequences=True)))

        # Flatten
        generator.add(keras.layers.Flatten())
        generator.add(keras.layers.Dense(self.trace_length))
        generator.add(keras.layers.Reshape((self.trace_length,1)))

        return generator

    def predict(self, noise):
        '''Function which uses the generator to generate noise'''
        return self.model.predict(noise)

    