from os import access
from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pywt


# Taken from this: https://github.com/fchollet/keras/pull/3677 since it is not implemented in Keras yet

class MinibatchDiscrimination(keras.layers.Layer):
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
        self.input_spec = [keras.engine.InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [keras.enginge.InputSpec(dtype=keras.backend.floatx(),
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
    def __init__(self, training = False):

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
        self.wavelet_levels = 2
        self.wavelet_trainable= False

        self.model = self.build_time_wavelet_fft_discriminator()
    
    def build_dcnn_discriminator(self):
        kwargs1 = {'kernel_size':9, 'padding':'same', 'kernel_initializer':'he_normal', 'strides':2}
        kwargs2 = {'kernel_size':9, 'padding':'same', 'kernel_initializer':'he_normal'}

        signal_input = keras.layers.Input(shape=(self.trace_length, 1), name = "signal_input")

        ### Time layers ###
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
        x = keras.layers.Flatten(name="time_out")(x)
        x = keras.layers.Flatten()(x)
    
        x = keras.layers.Dense(1)(x)
        

        critic = keras.models.Model(signal_input, x,name="critic")
        
        return critic

    def build_wavelet_discriminator(self):
            
        kwargs1 = {'kernel_size':9, 'padding':'same', 'kernel_initializer':'he_normal', 'strides':2}
        kwargs2 = {'kernel_size':9, 'padding':'same', 'kernel_initializer':'he_normal'}

        signal_input = keras.layers.Input(shape=(self.trace_length, 1), name = "signal_input")

        ## Wavelet layers ###

        # Wavelet Expansion
        approx_stack, detail_stack = self.make_wavelet_expansion(signal_input)

        # Create wavelet features
        features_list = []
        features_list.extend(detail_stack)
        features_list.append(approx_stack[-1])
        w_concat = keras.layers.Concatenate(axis=1, name="w_concat")(features_list)

        # Block 1
        w = keras.layers.Conv1D(filters=self.trace_length//32, **kwargs1)(w_concat)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate)(w)
        w = keras.layers.Conv1D(filters=self.trace_length//32, **kwargs2)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate)(w)

        # Block 2
        w = keras.layers.Conv1D(filters=self.trace_length//16, **kwargs1)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate)(w)
        w = keras.layers.Conv1D(filters=self.trace_length//16, **kwargs2)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate/2)(w)

        # Block 3
        w = keras.layers.Conv1D(filters=self.trace_length//8, **kwargs1)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate/2)(w)
        w = keras.layers.Conv1D(filters=self.trace_length//8, **kwargs2)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)

        w = keras.layers.Conv1D(filters=2, **kwargs1)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.Flatten(name="wavelet_out")(w)


        # Output layers
        out = keras.layers.Dense(1,activation='sigmoid')(w)
    

        critic = keras.models.Model(signal_input, out,name="critic")
        
        return critic

    def build_time_wavelet_discriminator(self):
            
        kwargs1 = {'kernel_size':9, 'padding':'same', 'kernel_initializer':'he_normal', 'strides':2}
        kwargs2 = {'kernel_size':9, 'padding':'same', 'kernel_initializer':'he_normal'}

        signal_input = keras.layers.Input(shape=(self.trace_length, 1), name = "signal_input")

        ### Time layers ###
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
        x = keras.layers.Flatten(name="time_out")(x)
        ########################

        ## Wavelet layers ###

        # Wavelet Expansion
        approx_stack, detail_stack = self.make_wavelet_expansion(signal_input)

        # Create wavelet features
        features_list = []
        features_list.extend(detail_stack)
        features_list.append(approx_stack[-1])
        w_concat = keras.layers.Concatenate(axis=1, name="w_concat")(features_list)


        w = keras.layers.Conv1D(filters=self.trace_length//32, **kwargs1)(w_concat)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate)(w)
        w = keras.layers.Conv1D(filters=self.trace_length//32, **kwargs2)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate)(w)

        # Block 2
        w = keras.layers.Conv1D(filters=self.trace_length//16, **kwargs1)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate)(w)
        w = keras.layers.Conv1D(filters=self.trace_length//16, **kwargs2)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate/2)(w)

        # Block 3
        w = keras.layers.Conv1D(filters=self.trace_length//8, **kwargs1)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate/2)(w)
        w = keras.layers.Conv1D(filters=self.trace_length//8, **kwargs2)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)

        w = keras.layers.Conv1D(filters=2, **kwargs1)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.Flatten(name="wavelet_out")(w)
        #############################

        # Concatante layer taking the output of the blocks above as input
        concat = keras.layers.Concatenate()([x,w])

        # Last layers
        out = keras.layers.Dense(1,activation='sigmoid')(w)

        # out = keras.layers.Dense(1)(x)
    

        critic = keras.models.Model(signal_input, out,name="critic")
        
        return critic

    def build_time_wavelet_fft_discriminator(self):
            
        kwargs1 = {'kernel_size':9, 'padding':'same', 'kernel_initializer':'he_normal', 'strides':2}
        kwargs2 = {'kernel_size':9, 'padding':'same', 'kernel_initializer':'he_normal'}

        signal_input = keras.layers.Input(shape=(self.trace_length, 1), name = "signal_input")

        ### Time layers ###
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
        x = keras.layers.Flatten(name="time_out")(x)
        ########################

        ## Wavelet layers ###

        # Wavelet Expansion
        approx_stack, detail_stack = self.make_wavelet_expansion(signal_input)

        # Create wavelet features
        features_list = []
        features_list.extend(detail_stack)
        features_list.append(approx_stack[-1])
        w_concat = keras.layers.Concatenate(axis=1, name="w_concat")(features_list)

        # Block 1
        w = keras.layers.Conv1D(filters=self.trace_length//32, **kwargs1)(w_concat)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate)(w)
        w = keras.layers.Conv1D(filters=self.trace_length//32, **kwargs2)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate)(w)

        # Block 2
        w = keras.layers.Conv1D(filters=self.trace_length//16, **kwargs1)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate)(w)
        w = keras.layers.Conv1D(filters=self.trace_length//16, **kwargs2)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate/2)(w)

        # Block 3
        w = keras.layers.Conv1D(filters=self.trace_length//8, **kwargs1)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)
        w = keras.layers.Dropout(self.dropout_rate/2)(w)
        w = keras.layers.Conv1D(filters=self.trace_length//8, **kwargs2)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.LayerNormalization()(w)

        w = keras.layers.Conv1D(filters=2, **kwargs1)(w)
        w = keras.layers.LeakyReLU()(w)
        w = keras.layers.Flatten(name="wavelet_out")(w)
        #############################

        ### FFT layers ###

        flat_input = keras.layers.Flatten()(signal_input)
        fft = keras.layers.Lambda(tf.signal.rfft)(flat_input)
        fft_abs = keras.layers.Lambda(keras.backend.abs)(fft)
        fft_abs = keras.layers.Reshape((-1,1), name='fft_abs')(fft_abs)

        # Block 1
        f = keras.layers.Conv1D(filters=self.trace_length//32, **kwargs1)(fft_abs)
        f = keras.layers.LeakyReLU()(f)
        f = keras.layers.LayerNormalization()(f)
        f = keras.layers.Dropout(self.dropout_rate)(f)
        f = keras.layers.Conv1D(filters=self.trace_length//32, **kwargs2)(f)
        f = keras.layers.LeakyReLU()(f)
        f = keras.layers.LayerNormalization()(f)
        f = keras.layers.Dropout(self.dropout_rate)(f)

        # Block 2
        f = keras.layers.Conv1D(filters=self.trace_length//16, **kwargs1)(f)
        f = keras.layers.LeakyReLU()(f)
        f = keras.layers.LayerNormalization()(f)
        f = keras.layers.Dropout(self.dropout_rate)(f)
        f = keras.layers.Conv1D(filters=self.trace_length//16, **kwargs2)(f)
        f = keras.layers.LeakyReLU()(f)
        f = keras.layers.LayerNormalization()(f)
        f = keras.layers.Dropout(self.dropout_rate/2)(f)

        # Block 3
        f = keras.layers.Conv1D(filters=self.trace_length//8, **kwargs1)(f)
        f = keras.layers.LeakyReLU()(f)
        f = keras.layers.LayerNormalization()(f)
        f = keras.layers.Dropout(self.dropout_rate/2)(f)
        f = keras.layers.Conv1D(filters=self.trace_length//8, **kwargs2)(f)
        f = keras.layers.LeakyReLU()(f)
        f = keras.layers.LayerNormalization()(f)

        f = keras.layers.Conv1D(filters=2, **kwargs1)(f)
        f = keras.layers.LeakyReLU()(f)
        f = keras.layers.Flatten(name="fft_out")(f)

        # Concatante layer taking the output of the blocks above as input
        concat = keras.layers.Concatenate()([x,w,f])

        # Last layers
        out = keras.layers.Dense(1,activation='sigmoid')(concat)

        # out = keras.layers.Dense(1)(x)
    

        critic = keras.models.Model(signal_input, out,name="critic")
        
        return critic
    
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