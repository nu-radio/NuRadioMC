from os import access
from time import time
from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pywt
from kapre import STFT, Magnitude, MagnitudeToDecibel


class MinibatchDiscrimination(keras.layers.Layer):
    # Taken from this: https://github.com/fchollet/keras/pull/3677 since it is not implemented in Keras yet
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

class Discriminator():
    def __init__(self, time_flag, fft_flag, wavelet_flag, mini_flag):

        '''Discriminator (critic) model used in the WGAN'''

        # Settings
        self.latent_size = 128
        self.channels = 1
        self.trace_length = 512

        # Layer config
        self.conv_activation = "relu"
        self.activation_function = "tanh"
        self.dropout_rate = 0.2

        # Sliding window
        self.sliding_window = 10
        self.moving_avg_window = 100

        # Wavelet
        self.wavelet_mother = "db7"
        self.wavelet_levels = 4
        self.wavelet_trainable= False

        # Layer settings
        self.time_flag = time_flag
        self.fft_flag = fft_flag 
        self.wavelet_flag = wavelet_flag
        self.mini_flag = mini_flag

        # self.model = self.build_critic()
        self.model = self.build_kapre()
        # self.test = self.build_critic()
        # self.wavelet = self.build_time_wavelet_discriminator()
        # self.time_wavelet = self.build_time_wavelet_discriminator()

    def build_critic(self):

         # Input
        signal_input = keras.layers.Input(shape=(self.trace_length, 1), name = "signal_input")
        # Flatten input
        flat_input = keras.layers.Flatten()(signal_input)

        output_layers = []


        if self.time_flag:

            # Build CNN blocks
            x = self.build_cnn_block(signal_input)

            # Flatten the output
            x = keras.layers.Flatten(name="time_out")(x)  

            output_layers.append(x)
 

        if self.wavelet_flag:

            # Wavelet Expansion
            approx_stack, detail_stack = self.make_wavelet_expansion(signal_input)

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

        critic = keras.models.Model(signal_input, out,name="critic")
        
        return critic 
    
    def build_cnn_block(self, signal_input):
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

    def make_wavelet_expansion(self, input_tensor):

        # Find filters for the current wavelet
        low_pass, high_pass  = pywt.Wavelet(self.wavelet_mother).filter_bank[:2]

        # Create arrays of filters
        low_pass_filter = np.array(low_pass)
        high_pass_filter = np.array(high_pass)

        # Set number of levels
        n_levels = self.wavelet_levels
        trainable=self.wavelet_trainable
        
        # Arguments for Conv
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

    def build_test(self):

        # Input
        signal_input = keras.layers.Input(shape=(self.trace_length, 1), name = "signal_input")

        # Flatted input
        flat_input = keras.layers.Flatten()(signal_input)

        # Perfrom STFT on the flat input signal
        stft = tf.keras.layers.Lambda(tf.signal.stft, arguments={"frame_length":self.trace_length//4, "frame_step":6})(flat_input)

        # Create Specgtrogram
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

        critic = keras.models.Model(signal_input, out,name="critic")

        return critic

    
    def build_kapre(self):
        """ Critic network """
        # Input

        critic = keras.models.Sequential()
        
        critic.add(keras.layers.Input(shape=(self.trace_length, 1), name = "signal_input"))

        # A STFT layer
        critic.add(STFT(n_fft=128, win_length=self.trace_length//4, hop_length=6,
                    window_name=None, pad_end=False,
                    input_data_format='channels_last', output_data_format='channels_last'))
        critic.add(Magnitude())
    #     critic.add(MagnitudeToDecibel())
        
        critic.add(keras.layers.Conv2D(16, kernel_size=3, strides=2, padding="valid"))
        critic.add(keras.layers.LeakyReLU())
        critic.add(keras.layers.Dropout(0.25))
        
        critic.add(keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same"))
        critic.add(keras.layers.LayerNormalization())
        critic.add(keras.layers.LeakyReLU())
        critic.add(keras.layers.Dropout(0.25))
        
        critic.add(keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
        critic.add(keras.layers.LayerNormalization())
        critic.add(keras.layers.LeakyReLU())
        critic.add(keras.layers.Dropout(0.25))
        
        critic.add(keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
        critic.add(keras.layers.LayerNormalization())
        critic.add(keras.layers.LeakyReLU())
        critic.add(keras.layers.Dropout(0.25))
        
        critic.add(keras.layers.Flatten())
        critic.add(keras.layers.Dense(1))
        
        
        return critic


















    # def build_dcnn_discriminator(self):

    #     # Input
    #     signal_input = keras.layers.Input(shape=(self.trace_length, 1), name = "signal_input")

    #     # Build CNN blocks
    #     x = self.build_cnn_block(signal_input)

    #     # Flatten the output and send through dense layer
    #     x = keras.layers.Flatten(name="time_out")(x)    
    #     x = keras.layers.Dense(1)(x)
        

    #     critic = keras.models.Model(signal_input, x,name="critic")
        
    #     return critic

    # def build_wavelet_discriminator(self):
            
    #     # Inputs
    #     signal_input = keras.layers.Input(shape=(self.trace_length, 1), name = "signal_input")

    #     # Wavelet Expansion
    #     approx_stack, detail_stack = self.make_wavelet_expansion(signal_input)

    #     # Create wavelet features
    #     features_list = []
    #     features_list.extend(detail_stack)
    #     features_list.append(approx_stack[-1])
    #     w_concat = keras.layers.Concatenate(axis=1, name="w_concat")(features_list)

    #     # Build CNN blocks
    #     w = self.build_cnn_block(w_concat)

    #     # Flatten the output and send through dense layer
    #     w = keras.layers.Flatten(name="wavelet_out")(w)
    #     out = keras.layers.Dense(1,activation='sigmoid')(w)
    

    #     critic = keras.models.Model(signal_input, out,name="critic")
        
    #     return critic

    # def build_time_wavelet_discriminator(self):
            
    #     # Input
    #     signal_input = keras.layers.Input(shape=(self.trace_length, 1), name = "signal_input")


    #     ### Time layers ###

    #     # Build CNN blocks
    #     x = self.build_cnn_block(signal_input)

    #     # Flatten the output and send through dense layer
    #     x = keras.layers.Flatten(name="time_out")(x)    
       
    #     ########################

    #     ### Wavelet layers ###

    #     # Wavelet Expansion
    #     approx_stack, detail_stack = self.make_wavelet_expansion(signal_input)

    #     # Create wavelet features
    #     features_list = []
    #     features_list.extend(detail_stack)
    #     features_list.append(approx_stack[-1])
    #     w_concat = keras.layers.Concatenate(axis=1, name="w_concat")(features_list)

    #     # Build CNN blocks
    #     w = self.build_cnn_block(w_concat)

    #     # Flatten
    #     w = keras.layers.Flatten(name="wavelet_out")(w)

    #     #############################

    #     # Concatante layer taking the output of the blocks above as input
    #     concat = keras.layers.Concatenate()([x,w])

    #     # Output layer
    #     out = keras.layers.Dense(1,activation='sigmoid')(concat)
    

    #     critic = keras.models.Model(signal_input, out,name="critic")
        
    #     return critic

    # def build_time_wavelet_fft_discriminator(self):
            
    #     # Input
    #     signal_input = keras.layers.Input(shape=(self.trace_length, 1), name = "signal_input")


    #     ### Time layers ###

    #     # Build CNN blocks
    #     x = self.build_cnn_block(signal_input)

    #     # Flatten the output and send through dense layer
    #     x = keras.layers.Flatten(name="time_out")(x)    
       
    #     ########################

    #     ### Wavelet layers ###

    #     # Wavelet Expansion
    #     approx_stack, detail_stack = self.make_wavelet_expansion(signal_input)

    #     # Create wavelet features
    #     features_list = []
    #     features_list.extend(detail_stack)
    #     features_list.append(approx_stack[-1])
    #     w_concat = keras.layers.Concatenate(axis=1, name="w_concat")(features_list)

    #     # Build CNN blocks
    #     w = self.build_cnn_block(w_concat)

    #     # Flatten
    #     w = keras.layers.Flatten(name="wavelet_out")(w)
    #     #############################

    #     ### FFT layers ###

    #     # Create FFT input
    #     flat_input = keras.layers.Flatten()(signal_input)
    #     fft = keras.layers.Lambda(tf.signal.rfft)(flat_input)
    #     fft_abs = keras.layers.Lambda(keras.backend.abs)(fft)
    #     fft_abs = keras.layers.Reshape((-1,1), name='fft_abs')(fft_abs)

    #     f = self.build_cnn_block(fft_abs)
    #     f = keras.layers.Flatten(name="fft_out")(f)

    #     #########################


    #     # MiniBatchDiscrimination
    #     mini = MinibatchDiscrimination(10,3)(flat_input)

    #     # Concatante layer taking the output of the blocks above as input
    #     concat = keras.layers.Concatenate()([x,w,f, mini])

    #     # Last layers
    #     out = keras.layers.Dense(1,activation='sigmoid')(concat)

    #     critic = keras.models.Model(signal_input, out,name="critic")
        
    #     return critic

    
    

    def make_wavelet_expansion(self, input_tensor):

        # Find filters for the current wavelet
        low_pass, high_pass  = pywt.Wavelet(self.wavelet_mother).filter_bank[:2]

        # Create arrays of filters
        low_pass_filter = np.array(low_pass)
        high_pass_filter = np.array(high_pass)

        # Set number of levels
        n_levels = self.wavelet_levels
        trainable=self.wavelet_trainable
        
        # Arguments for Conv
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