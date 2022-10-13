from os import access
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pywt
from kapre import STFT, Magnitude
from keras import backend as K
from wavetf import WaveTFFactory
from tf_cwt import Wavelet1D, Scaler, RGBStack



class Discriminator():
    def __init__(self, trace_length, latent_size, time_flag=True, fft_flag=True, wavelet_flag=True, envelope_flag=False, mini_flag=False):

        '''Discriminator (critic) class used in the WGAN'''

        # Settings for latent size and trace length
        self.latent_size = latent_size
        self.trace_length = trace_length

        # Architecture config
        self.dropout_rate = 0.2
        self.moving_avg_window = 100

        # Wavelet
        self.wavelet_mother = "db7"
        self.wavelet_levels = 2
        self.wavelet_trainable= False

        # Layer settings 
        self.time_flag = time_flag
        self.fft_flag = fft_flag 
        self.wavelet_flag = wavelet_flag
        self.mini_flag = mini_flag
        self.envelope_flag = envelope_flag

        # Define which Discriminator architecture to use
        self.model = self.build_stft_kapre()

    
    def build_time_fft_wavelet(self):

        """ Critic network using time, FFT and Wavelet layers. Inspired by: https://github.com/larocs/EMG-GAN """
        
        # Input
        trace_input = keras.layers.Input(shape=(self.trace_length, 1), name = "trace_input")

        # Flatten input
        flat_input = keras.layers.Flatten()(trace_input)

        output_layers = []


        if self.time_flag:

            # Build CNN blocks
            x = self.build_cnn_block(trace_input)

            # Flatten the output
            x = keras.layers.Flatten(name="time_out")(x)  

            output_layers.append(x)
 

        if self.wavelet_flag:

            # Wavelet Expansion
            approx_stack, detail_stack = self.make_wavelet_expansion(trace_input)

            # Create wavelet features
            features_list = []
            features_list.extend(detail_stack)
            features_list.append(approx_stack[-1])
            w_concat = keras.layers.Concatenate(axis=1, name="w_concat")(features_list)

            # Build CNN blocks
            w = self.build_cnn_block(w_concat)

            # Flatten
            w = keras.layers.Flatten(name="wavelet_out")(w)

            output_layers.append(w)

        if self.fft_flag:

            # Create FFT input
            fft = keras.layers.Lambda(tf.signal.rfft)(flat_input)
            fft_abs = keras.layers.Lambda(keras.backend.abs)(fft)
            fft_abs = keras.layers.Reshape((-1,1), name='fft_abs')(fft_abs)

            # Build CNN blocks
            f = self.build_cnn_block(fft_abs)

            # Flatten
            f = keras.layers.Flatten(name="fft_out")(f)

            output_layers.append(f)
        
        if self.envelope_flag:

            # Create envelope of FFT layer
            e = keras.layers.Lambda(self.envelopes)(trace_input)
            e = keras.layers.Flatten()(e)
            e = keras.layers.Lambda(tf.signal.rfft)(e)
            e = keras.layers.Lambda(keras.backend.abs)(e)
            e = keras.layers.Reshape((-1,1), name = "envelope")(e)

            e = self.build_cnn_block(e)

            e = keras.layers.Flatten(name="envelope_out")(e)

            output_layers.append(e)


        if self.mini_flag:
            # MiniBatchDiscrimination
            mini = MinibatchDiscrimination(10,3)(flat_input)
            output_layers.append(mini)


        if len(output_layers) > 1:
            # Concatante layer taking the output of the blocks above as input
            last_layer = keras.layers.Concatenate()(output_layers)
        else:
            last_layer = output_layers[0]


        # Last layers
        out = keras.layers.Dense(1,activation='sigmoid')(last_layer)

        critic = keras.models.Model(trace_input, out,name="critic")
        
        return critic 
    
    def build_stft(self):

        """ Critic network using Short Time Fourier Transform layer and 2D convolutional layers """

        # Input
        trace_input = keras.layers.Input(shape=(self.trace_length, 1), name = "trace_input")

        # Flatted input
        flat_input = keras.layers.Flatten()(trace_input)

        # Perfrom STFT on the flat input signal
        stft = tf.keras.layers.Lambda(tf.signal.stft, arguments={"frame_length":self.trace_length//4, "frame_step":6})(flat_input)

        # Create Spectrogram
        spectogram = tf.keras.layers.Lambda(tf.abs)(stft)

        # Expand dimensions to be able to use 2D convolutional layers
        spectogram = tf.expand_dims(spectogram, -1)

        # 2D Convolutional layer block
        x = keras.layers.Conv2D(16, kernel_size=3, strides=2, padding="valid")(spectogram)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same")(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.25)(x)

        # Flatten and Dense output
        x = keras.layers.Flatten()(x)
        out = keras.layers.Dense(1,activation='sigmoid')(x)

        critic = keras.models.Model(trace_input, out,name="critic")

        return critic

    def build_stft_kapre(self):

        """ Critic network using Kapre's Short Time Fourier Transform layers and 2D convolutional layers """

        # Arguments for the convolutional layers
        kwargs = {'kernel_size':3 ,'kernel_initializer':'he_normal', 'strides':2}

        # Create model
        critic = keras.models.Sequential()
        
        # Input
        critic.add(keras.layers.Input(shape=(self.trace_length, 1), name = "trace_input"))

        # 512 sample model
        if self.trace_length == 512:

            # STFT layer
            critic.add(STFT(n_fft=128, win_length=self.trace_length//4, hop_length=6,
                        window_name=None, pad_end=False,
                        input_data_format='channels_last', output_data_format='channels_last'))

            # Turn the STFT output to spectrogram            
            critic.add(Magnitude())
            
            # Conv2D blockss
            critic.add(keras.layers.Conv2D(16, padding="valid", **kwargs))
            critic.add(keras.layers.LeakyReLU())
            critic.add(keras.layers.Dropout(self.dropout_rate))
            
            critic.add(keras.layers.Conv2D(32, padding="same", **kwargs))
            critic.add(keras.layers.LayerNormalization())
            critic.add(keras.layers.LeakyReLU())
            critic.add(keras.layers.Dropout(self.dropout_rate))
            
            critic.add(keras.layers.Conv2D(64, padding="same", **kwargs))
            critic.add(keras.layers.LayerNormalization())
            critic.add(keras.layers.LeakyReLU())
            critic.add(keras.layers.Dropout(self.dropout_rate//2))
            
            critic.add(keras.layers.Conv2D(128, padding="same", **kwargs))
            critic.add(keras.layers.LayerNormalization())
            critic.add(keras.layers.LeakyReLU())
            
            critic.add(keras.layers.Flatten())
            critic.add(keras.layers.Dense(1))
        
        # 2048 sample model. Not optimized and tested thorougly.
        if self.trace_length == 2048:

            # STFT layer
            critic.add(STFT(n_fft=128*4, win_length=512, hop_length=6,
                        window_name=None, pad_end=False,
                        input_data_format='channels_last', output_data_format='channels_last'))

            # Turn the STFT output to spectrogram            
            critic.add(Magnitude())
            
            # Conv2D blocks
            critic.add(keras.layers.Conv2D(16, padding="valid", **kwargs))
            critic.add(keras.layers.LeakyReLU())
            critic.add(keras.layers.Dropout(0.2))
            
            critic.add(keras.layers.Conv2D(32, padding="same", **kwargs))
            critic.add(keras.layers.LayerNormalization())
            critic.add(keras.layers.LeakyReLU())
            critic.add(keras.layers.Dropout(0.2))
            
            critic.add(keras.layers.Conv2D(64, padding="same", **kwargs))
            critic.add(keras.layers.LayerNormalization())
            critic.add(keras.layers.LeakyReLU())
            critic.add(keras.layers.Dropout(0.2//2))
            
            critic.add(keras.layers.Conv2D(32, padding="same", **kwargs))
            critic.add(keras.layers.LayerNormalization())
            critic.add(keras.layers.LeakyReLU())
            
            critic.add(keras.layers.Conv2D(16, padding="same", **kwargs))
            critic.add(keras.layers.LeakyReLU())
            critic.add(keras.layers.Dropout(0.2))
            
            critic.add(keras.layers.Flatten())
            critic.add(keras.layers.Dense(1))
            
        return critic
    
    def build_lstm(self):

        '''
            LSTM Discriminator inspired by https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8942842
            
            Doesn't work particullary well and training is significantly slower than other models. 
        
        '''
        
        # Arguments for the convultional layers
        kwargs = {'kernel_size':3, 'padding':'same', 'kernel_initializer':'he_normal', 'strides':2}


        # Define model
        critic = keras.models.Sequential()
        
        # Input
        critic.add(keras.layers.Input(shape=(self.trace_length, 1), name = "trace_input"))

        # Dense
        critic.add(keras.layers.Dense(100))
        critic.add(keras.layers.LeakyReLU())

        critic.add(keras.layers.Bidirectional(keras.layers.LSTM(16, return_sequences=True)))


        # Convolutional layers
        critic.add(keras.layers.Conv1D(filters=16, **kwargs))
        critic.add(keras.layers.LeakyReLU())
        critic.add(keras.layers.LayerNormalization())
        critic.add(keras.layers.Dropout(self.dropout_rate))

        critic.add(keras.layers.Conv1D(filters=32, **kwargs))
        critic.add(keras.layers.LeakyReLU())
        critic.add(keras.layers.LayerNormalization())
        critic.add(keras.layers.Dropout(self.dropout_rate))

        critic.add(keras.layers.Conv1D(filters=64, **kwargs))
        critic.add(keras.layers.LeakyReLU())
        critic.add(keras.layers.LayerNormalization())
        critic.add(keras.layers.Dropout(self.dropout_rate))

        # Flatten and Dense output
        critic.add(keras.layers.Flatten())
        critic.add(keras.layers.Dense(1,activation='sigmoid'))

        return critic

    def build_wavetf(self):

        ''' Critic using the WaveTFFactory (https://github.com/fversaci/WaveTF) implementation '''

        # Input
        trace_input = keras.layers.Input(shape=(self.trace_length, 1), name = "trace_input")

        # Compute 4 level of Wavelet transform
        wave_1 = WaveTFFactory.build(kernel_type='db2', dim = 1)(trace_input)
        wave_2 = WaveTFFactory.build(kernel_type='db2', dim = 1)(wave_1)
        wave_3 = WaveTFFactory.build(kernel_type='db2', dim = 1)(wave_2)
        wave_4 = WaveTFFactory.build(kernel_type='db2', dim = 1)(wave_3)

        wavelets = [wave_1, wave_2, wave_3, wave_4]

        # for wave in wavelets:
        #     wave = keras.layers.BatchNormalization()(wave)

        
        kinit ='glorot_normal' # 'he_normal'

        def rep_conv(cnn, scale = 1) :
            for i in range(2) :
                cnn = keras.layers.Conv1D(scale * 4, kernel_size=3, activation = 'relu', padding = 'same',
                            kernel_initializer = kinit)(cnn)
            return cnn

        def pool_down(cnn, mul):
            cnn = keras.layers.Conv1D(mul * 4, kernel_size=3, activation = 'relu', padding = 'same',
                        kernel_initializer = kinit, strides=2)(cnn)
            return (cnn)
        
        cnn = rep_conv(trace_input, 1)

        for l in range(len(wavelets)) :
            cnn = pool_down(cnn, 2**(l+1))
            cnn = rep_conv(cnn, 2**(l+1))
            cnn = keras.layers.Concatenate(axis=2)([cnn, wavelets[l]])

        # output
        cnn = keras.layers.Conv1D(2048, kernel_size=3)(cnn)
        cnn = keras.layers.Activation('relu')(cnn)
        cnn = keras.layers.GlobalAveragePooling1D()(cnn)
        

        # Flatten and Dense output
        x = keras.layers.Flatten()(cnn)
        out = keras.layers.Dense(1,activation='sigmoid')(x)

        critic = keras.models.Model(trace_input, out,name="critic")

        return critic
    
    def build_wavelet1d(self):
        '''Critic using the Wavelet1D (https://www.kaggle.com/code/mistag/wavelet1d-custom-keras-wavelet-transform-layer/notebook)
            layer. Does not train at all and I don't know why. Probably something wrong with Wavelet1D.
        '''
        # Input
        trace_input = keras.layers.Input(shape=(self.trace_length, 1), name = "trace_input")
        print(trace_input.shape)
        
        # Flatten input
        flatten = keras.layers.Flatten()(trace_input)

        wavelet = Wavelet1D(nv=68, sr=2048., flow=10, fhigh=100, batch_size=64)(flatten)
        wavelet = Scaler(upper=255.)(wavelet)

        # Expand dimensions to be able to use 2D convolutional layers
        wavelet = tf.expand_dims(wavelet, -1)

        # 2D Convolutional layer block
        x = keras.layers.ZeroPadding2D(padding=(1, 0))(wavelet)

        x = keras.layers.Conv2D(4, kernel_size=3, strides=2, padding="same")(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(8, kernel_size=3, strides=2, padding="same")(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(16, kernel_size=3, strides=2, padding="same")(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same")(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.25)(x)       

        # Flatten and Dense output
        x = keras.layers.Flatten()(x)
        out = keras.layers.Dense(1,activation='sigmoid')(x)

        critic = keras.models.Model(trace_input, out,name="critic")

        return critic
    
    def make_wavelet_expansion(self, input_tensor):

        '''Function creating the Wavelet block for build_time_fft_wavelet. Inspired by: https://github.com/larocs/EMG-GAN '''

        # Find filters for the current wavelet
        low_pass, high_pass  = pywt.Wavelet(self.wavelet_mother).filter_bank[:2]

        # Create arrays of filters
        low_pass_filter = np.array(low_pass)
        high_pass_filter = np.array(high_pass)

        # Set number of levels
        n_levels = self.wavelet_levels
        trainable=self.wavelet_trainable
        
        # Arguments for the convolutional layer
        kwargs = {
            "filters":1,
            "kernel_size":len(low_pass),
            "strides":2,     
            "use_bias":False, 
            "padding":"same", 
            "trainable":trainable,
        }

        # Create list for approxmation and detail coeff
        approximation_coefficients = []
        detail_coefficients = []

        last_approximant = input_tensor

        # Loop over levels
        for i in range(n_levels):
            
            # Conv for approximation layer
            a_n = keras.layers.Conv1D(
                kernel_initializer=keras.initializers.Constant(low_pass_filter.reshape((-1, 1))),name="low_pass_{}".format(i),**kwargs)(last_approximant)
            
            # Conv for detailed layer
            d_n = keras.layers.Conv1D(kernel_initializer=keras.initializers.Constant(high_pass_filter.reshape((-1, 1))),name="high_pass_{}".format(i),**kwargs,)(last_approximant)

            # Add to lists
            detail_coefficients.append(d_n)
            approximation_coefficients.append(a_n)

            # Update last_approximant
            last_approximant = a_n

        return approximation_coefficients, detail_coefficients

    def build_cnn_block(self, signal_input):

        '''Deep convolutional block '''

        # Arguments used in the convolutional layers
        kwargs1 = {'kernel_size':9, 'padding':'same', 'kernel_initializer':'he_normal', 'strides':2}
        kwargs2 = {'kernel_size':9, 'padding':'same', 'kernel_initializer':'he_normal'}

        # Block 1
        x = keras.layers.Conv1D(filters=self.trace_length//32, **kwargs1)(signal_input)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        x = keras.layers.Conv1D(filters=self.trace_length//32, **kwargs2)(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)

        # Block 2
        x = keras.layers.Conv1D(filters=self.trace_length//16, **kwargs1)(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        x = keras.layers.Conv1D(filters=self.trace_length//16, **kwargs2)(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(self.dropout_rate/2)(x)

        # Block 3
        x = keras.layers.Conv1D(filters=self.trace_length//8, **kwargs1)(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dropout(self.dropout_rate/2)(x)
        x = keras.layers.Conv1D(filters=self.trace_length//8, **kwargs2)(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.LayerNormalization()(x)

        x = keras.layers.Conv1D(filters=2, **kwargs1)(x)
        x = keras.layers.LeakyReLU()(x)

        return x
    
    def envelopes(self, input):
        '''
        Function to calculate the envelope moving average.
        Implemented for the build_time_fft_wavelet but doesn't improve the performance.
        '''
        envelope = tf.signal.frame(
            input,
            self.moving_avg_window,
            1,#steps
            pad_end=True,
            pad_value=0,
            axis=1,
            name='envelope_moving_average'
        )
        envelope_reshaped = K.reshape(envelope,(-1,self.trace_length,self.moving_avg_window))
        envelope_mean = K.mean(envelope_reshaped, axis=2, keepdims=True)
        return envelope_mean

class MinibatchDiscrimination(keras.layers.Layer):
    # Taken from this: https://github.com/fchollet/keras/pull/3677 since it is not implemented in Keras yet
    # Implemented for the build_time_fft_wavelet but doesn't improve the performance
        
    """Concatenates to each sample information about how different the input
    features for that sample are from features of other samples in the same
    minibatch, as described in Salimans et. al. (2016). Useful for preventing
    GANs from collapsing to a single output. When using this layer, generated
    samples and reference samples should be in separate batches.
    # Example
    ```python
        # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
        # with 64 output filters
        model = Sequential()
        model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
        # now model.output_shape == (None, 10, 64)
        # flatten the output so it can be fed into a minibatch discrimination layer
        model.add(Flatten())
        # now model.output_shape == (None, 640)
        # add the minibatch discrimination layer
        model.add(MinibatchDiscrimination(5, 3))
        # now model.output_shape = (None, 645)
    ```
    # Arguments
        nb_kernels: Number of discrimination kernels to use
            (dimensionality concatenated to output).
        kernel_dim: The dimensionality of the space where closeness of samples
            is calculated.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        weights: list of numpy arrays to set as initial weights.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(samples, input_dim + nb_kernels)`.
    # References
        - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
    """

    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = keras.initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        self.W_constraint = keras.constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [keras.layers.InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [keras.layers.InputSpec(dtype=keras.backend.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
            initializer=self.init,
            name='kernel',
            regularizer=self.W_regularizer,
            trainable=True,
            constraint=self.W_constraint)

        # Set built to true.
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = keras.backend.reshape(keras.backend.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = keras.backend.expand_dims(activation, 3) - keras.backend.expand_dims(keras.backend.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = keras.backend.sum(keras.backend.abs(diffs), axis=2)
        minibatch_features = keras.backend.sum(keras.backend.exp(-abs_diffs), axis=2)
        return keras.backend.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1]+self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
