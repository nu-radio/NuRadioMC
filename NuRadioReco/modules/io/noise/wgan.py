# Imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
layers = keras.layers
tf.compat.v1.disable_eager_execution()  # gp loss won't work with eager
from functools import partial
from generator import Generator
from discriminator import Discriminator
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("analyze.py"))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("current_noise.npy"))))
sys.path.insert(1, '/lustre/fs22/group/radio/dhjelm/')
import analyze


class WGAN():
    def __init__(self, trace_length, time_flag=True, fft_flag=True, wavelet_flag=True, envelope_flag=False, mini_flag=False):

        '''Wasserstein GAN with Gradient Penalty'''

        # Settings for latent size and trace_length
        self.latent_size = 128
        self.trace_length = trace_length
        
        # Training settings 
        self.batch_size = 64
        self.learning_rate = 0.0001

        # Discriminator settings
        self.time_flag = time_flag
        self.fft_flag = fft_flag 
        self.wavelet_flag = wavelet_flag
        self.mini_flag = mini_flag
        self.envelope_flag = envelope_flag

        #-------------------------------
        # Create the networks
        #-------------------------------

        # Create generator
        self.generator = Generator(trace_length = self.trace_length, latent_size = self.latent_size).model
        self.generator.summary()
       

        # Create critic
        self.critic = Discriminator(self.trace_length, self.latent_size, self.time_flag, self.fft_flag, self.wavelet_flag, self.envelope_flag, self.mini_flag).model
        self.critic.summary()
        
        #-------------------------------
        # Construct Computational Graph
        #         for the GAN
        #-------------------------------

        # Freeze the critic during the generator training and unfreeze the generator during the generator training
        self.make_trainable(self.critic, False) 
        self.make_trainable(self.generator, True)

        # Stack the generator on top of the critic and finiliaze the training pipeline of the generator
        gen_input = self.generator.inputs
        self.generator_training = keras.models.Model(gen_input, self.critic(self.generator(gen_input)))

        # Compile generator for training using the Wasserstein loss as loss function
        self.generator_training.compile(keras.optimizers.Adam(self.learning_rate, beta_1=0.5, beta_2=0.9, decay=0.0), loss=[self.wasserstein_loss])

        self.make_trainable(self.critic, True)  # unfreeze the critic during the critic training
        self.make_trainable(self.generator, False)  # freeze the generator during the critic training

        g_out = self.generator(self.generator.inputs)
        critic_out_fake_samples = self.critic(g_out)
        critic_out_data_samples = self.critic(self.critic.inputs)
        averaged_batch = UniformLineSampler(self.batch_size)([g_out, self.critic.inputs[0]])
        averaged_batch_out = self.critic(averaged_batch)
        self.critic_training = keras.models.Model(inputs=[self.generator.inputs, self.critic.inputs], outputs=[critic_out_fake_samples, critic_out_data_samples, averaged_batch_out])

        # Construct the gradient penalty
        gradient_penalty_weight = 1
        gradient_penalty = partial(self.gradient_penalty_loss, averaged_batch=averaged_batch, penalty_weight=gradient_penalty_weight)  
        gradient_penalty.__name__ = 'gradient_penalty'

        # Compile critic with gradient penalty
        self.critic_training.compile(keras.optimizers.Adam(self.learning_rate/2, beta_1=0.5, beta_2=0.9, decay=0.0), loss=[self.wasserstein_loss, self.wasserstein_loss, gradient_penalty])

    def train(self, data, epochs = 50, monitor = True):

        '''
        Function training the network
        
        Args: 
            data: Data to be trained on
            epochs: How many epochs the network should be trained
            monitor: Turns on/off the plotting every 5th epoch

        Outputs:
            generator_loss: A list of generator loss
            critic_loss: A list of critic loss
        '''


        # Create arrays for generator and critic loss
        generator_loss = []
        critic_loss = []

        # Keras throws an error when calculating a loss without having a label -> needed for using the gradient penalty loss
        positive_y = np.ones(self.batch_size)
        negative_y = -positive_y
        dummy = np.zeros(self.batch_size) 


        # Calculate number of iterations per epoch
        iterations_per_epoch = len(data)//(self.batch_size)
        
        for epoch in range(epochs):
    
            print("Epoch: ", epoch)
            
            for iteration in range(iterations_per_epoch):
                
                #-------------------------------
                # Critic training
                #-------------------------------

                # Pick data in batches 
                bunch=data[self.batch_size*(iteration):self.batch_size*(iteration+1)]
                bunch = np.expand_dims(bunch, axis=-1)  

                # Generate noise
                noise_batch = np.random.randn(len(bunch), self.latent_size)
                noise_batch = np.expand_dims(noise_batch, axis=-1) 

                # Train critic
                critic_loss.append(self.critic_training.train_on_batch([noise_batch, bunch], [negative_y, positive_y, dummy]))
                
                # Train the generator only every fifth iteration, i.e train the critic five times more than the generator
                if iteration & 5 == 0:

                    #-------------------------------
                    # Generator training
                    #-------------------------------

                    # Generate noise batch for generator
                    noise_batch = np.random.randn(self.batch_size, self.latent_size)
                    noise_batch = np.expand_dims(noise_batch, axis=-1) 

                    # Train the generator
                    generator_loss.append(self.generator_training.train_on_batch([noise_batch], [positive_y]))  


            
            # Loss printing and plotting
            if monitor and epoch % 5 == 0:
                
                # Loss
                print("\n")
                print("Critic loss:", critic_loss[-1])
                print("Generator loss:", generator_loss[-1])
                
                # Generate noise
                noise = np.random.randn(self.batch_size, self.latent_size)
                noise = np.expand_dims(noise, axis=-1) 
                generated_signals = self.generator.predict_on_batch(noise)
                generated_signals = generated_signals[:,:,0]

                # Analyze the generator's performance
                self.analyze_generator(data)


        return  generator_loss, critic_loss

    def plot_loss(self, generator_loss, critic_loss):
        '''Plot the loss for generator and critic as a function of iterations '''

        # Critic loss
        critic_loss = np.array(critic_loss)
        plt.subplots(1, figsize=(10, 5))
        plt.plot(np.arange(len(critic_loss)), critic_loss[:,0], color='red', markersize=12, label=r'Total')
        plt.plot(np.arange(len(critic_loss)), critic_loss[:,1] + critic_loss[:, 2], color='green', label=r'Wasserstein', linestyle='dashed')
        plt.plot(np.arange(len(critic_loss)), critic_loss[:, 3], color='royalblue', markersize=12, label=r'Gradient penalty', linestyle='dashed')
        plt.legend(loc='upper right')
        plt.xlabel(r'Iterations')
        plt.ylabel(r'Critic Loss')

        # Generator loss
        generator_loss = np.array(generator_loss)
        plt.subplots(1, figsize=(10, 5))
        plt.plot(np.arange(len(generator_loss)), generator_loss, color='red', markersize=12, label=r'Total')
        plt.legend(loc='upper right')
        plt.xlabel(r'Iterations')
        plt.ylabel(r'Loss')

    def analyze_generator(self, data):

        '''
        Analyzes the generator's ability to generate realistic noise
        
        Args: 
            data: Data to be trained on

        Outputs:
            Prints metrics and display plots
        '''

        # Generate noise
        noise = np.random.randn(len(data), self.latent_size)
        noise = np.expand_dims(noise, axis=-1) 

        # Generate traces using generator
        generated_traces = self.generator.predict_on_batch(noise)
        generated_traces = generated_traces[:,:,0]

        # Print metrics
        analyze.mean_std(data, generated_traces)

        # FFT MSE
        analyze.fft_mse(data, generated_traces)

        # Plot distributions
        analyze.plot_distributions(data, generated_traces)

        # Plot average, median and quantile frequencies
        analyze.avg_med_quantile_freq(data, generated_traces)
       
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
        The 1-Lipschitz constraint of improved selfs is enforced by adding a term that penalizes a gradient norm in the critic unequal to 1."""
        
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
    
    def plot_critic(self):
        '''Plots the critic using Keras utils'''
        return keras.utils.plot_model(self.critic, show_layer_names=False)
    
    def plot_generator(self):
        '''Plots the generator using Keras utils'''
        return keras.utils.plot_model(self.generator, show_layer_names=False)



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