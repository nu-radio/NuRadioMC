import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
layers = keras.layers
tf.compat.v1.disable_eager_execution()  # gp loss won't work with eager
from functools import partial
from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from NuRadioReco.framework import base_trace
from generator import Generator
from discriminator import Discriminator


class WGAN():
    def __init__(self, training = False):

        '''Wasserstein GAN with Gradient Penalty'''

        self.latent_size = 128
        self.channels = 1
        self.conv_activation = "relu"
        self.activation_function = "tanh"
        self.trace_length = 512
        self.shape = (self.trace_length, self.channels)
        self.sliding_window = 10
        self.batch_size = 64
        self.dropout_rate = 0.2

        # Create generator
        self.generator = Generator().model
        # self.generator.summary()
        # print(self.generator)
       

        # Create critic
        self.critic = Discriminator().model
        self.critic.summary()

        #-------------------------------
        # Construct Computational Graph
        #         for the GAN
        #-------------------------------

        # Freeze the critic during the generator training and unfreeze the generator during the generator training
        self.make_trainable(self.critic, False) 
        self.make_trainable(self.generator, True)

        # Stack the generator o top of the critic and finiliaze the training pipeline of the generator
        gen_input = self.generator.inputs
        self.generator_training = keras.models.Model(gen_input, self.critic(self.generator(gen_input)))
        # self.generator_training.summary()
        # keras.utils.plot_model(generator_training, show_shapes=True)

        # Compile generator for training using the Wasserstein loss as loss function
        self.generator_training.compile(keras.optimizers.Adam(
            0.0001, beta_1=0.5, beta_2=0.9, decay=0.0), loss=[self.wasserstein_loss])

        self.make_trainable(self.critic, True)  # unfreeze the critic during the critic training
        self.make_trainable(self.generator, False)  # freeze the generator during the critic training

        g_out = self.generator(self.generator.inputs)
        critic_out_fake_samples = self.critic(g_out)
        critic_out_data_samples = self.critic(self.critic.inputs)
        averaged_batch = UniformLineSampler(self.batch_size)([g_out, self.critic.inputs[0]])
        averaged_batch_out = self.critic(averaged_batch)
        self.critic_training = keras.models.Model(inputs=[self.generator.inputs, self.critic.inputs], outputs=[critic_out_fake_samples, critic_out_data_samples, averaged_batch_out])
        # self.critic_training.summary()

        # Construct the gradient penalty
        gradient_penalty_weight = 1
        gradient_penalty = partial(self.gradient_penalty_loss, averaged_batch=averaged_batch, penalty_weight=gradient_penalty_weight)  
        gradient_penalty.__name__ = 'gradient_penalty'

        # Compile critic with gradient penalty
        self.critic_training.compile(keras.optimizers.Adam(0.00005, beta_1=0.5, beta_2=0.9, decay=0.0), loss=[self.wasserstein_loss, self.wasserstein_loss, gradient_penalty])



    def make_trainable(self, model, trainable):
        ''' Freezes/unfreezes the weights in the given model '''
        for layer in model.layers:

            if type(layer) is layers.BatchNormalization:
                layer.trainable = True
            else:
                layer.trainable = trainable

    def wasserstein_loss(self, y_true, y_pred):
        """Calculates the Wasserstein loss - critic maximises the distance between its output for real and generated samples.
        To achieve this generated samples have the label -1 and real samples the label 1. Multiplying the outputs by the labels results to the wasserstein loss via the Kantorovich-Rubinstein duality"""
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_batch, penalty_weight):
        """Calculates the gradient penalty.
        The 1-Lipschitz constraint of improved WGANs is enforced by adding a term that penalizes a gradient norm in the critic unequal to 1."""
        
        # Gradients
        gradients = K.gradients(y_pred, averaged_batch)[0]
        
        # Gradients sqruared
        gradients_sqr = K.square(gradients)
        
        # Sum over the rows
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        
        # Sqrt
        gradient_l2_norm =  K.sqrt(gradients_sqr_sum) 
                                
        # Compute penalty_weight * (1 - ||grad||)^2 still for each single sample                        
        gradient_penalty = penalty_weight * K.square(1 - gradient_l2_norm)
        
        return K.mean(gradient_penalty)
        



# To obtain the Wasserstein distance, we have to use the gradient penalty to enforce the Lipschitz constraint. 
# Therefore, we need to design a layer that samples on straight lines between reals and fakes samples.
class UniformLineSampler(tf.keras.layers.Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        weights = K.random_uniform((self.batch_size, 1, 1))
        return(weights * inputs[0]) + ((1 - weights) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

if __name__ == '__main__':
    model = WGAN()
    print(model)