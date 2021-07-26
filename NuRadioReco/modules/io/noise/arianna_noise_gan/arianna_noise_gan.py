#!/usr/bin/env pygpu
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
'''from NuRadioReco.modules.io.noise import noiseReaderUproot '''#only import noiseReader if you use it
import utils
from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
tf.compat.v1.disable_eager_execution()  # gp loss won't work with eager
layers = keras.layers


# basic training parameter
EPOCHS = 10
GRADIENT_PENALTY_WEIGHT = 10
BATCH_SIZE = 200
NCR = 5
latent_size = 256


#N=100
#data=np.random.normal(size=(N,256))
# load trainings data
N,data = utils.ReadInData('C:/users/Simon Hillmann/OneDrive/Dokumente/Uni/Bachelor/station_32_run_00225.root')
nsamples = N*4#4 traces per event


data=np.concatenate([data[i,:,:] for i in range(len(data[:,0,0]))])#write all traves into a 1d-array
np.random.shuffle(data)#randomize order
#print(np.shape(data))




g = utils.generator_model(latent_size)

g.summary()


critic = utils.critic_model(latent_size)

critic.summary()


gen_input = g.inputs

generator_training = keras.models.Model(gen_input, critic(g(gen_input)))

generator_training.summary()

keras.utils.plot_model(generator_training, show_shapes=True)


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss - critic maximises the distance between its output for real and generated samples.
    To achieve this generated samples have the label -1 and real samples the label 1. Multiplying the outputs by the labels results to the wasserstein loss via the Kantorovich-Rubinstein duality"""
    return K.mean(y_true * y_pred)


generator_training.compile(keras.optimizers.Adam(
    0.0001, beta_1=0.5, beta_2=0.9, decay=0.0), loss=[wasserstein_loss])


utils.make_trainable(g, False)  # freeze the generator during the critic training
utils.make_trainable(critic, True)  # unfreeze the critic during the critic training

g_out = g(g.inputs)
critic_out_fake_samples = critic(g_out)
critic_out_data_samples = critic(critic.inputs)
averaged_batch = utils.UniformLineSampler(BATCH_SIZE)([g_out, critic.inputs[0]])
averaged_batch_out = critic(averaged_batch)

critic_training = keras.models.Model(inputs=[g.inputs, critic.inputs], outputs=[
                                     critic_out_fake_samples, critic_out_data_samples, averaged_batch_out])

critic_training.summary()


def gradient_penalty_loss(y_true, y_pred, averaged_batch, penalty_weight):
    """Calculates the gradient penalty.
    The 1-Lipschitz constraint of improved WGANs is enforced by adding a term that penalizes a gradient norm in the critic unequal to 1."""
    gradients = K.gradients(y_pred, averaged_batch)
    gradients_sqr_sum = K.sum(K.square(gradients)[0], axis=(1))
    gradient_penalty = penalty_weight * K.square(1 - K.sqrt(gradients_sqr_sum))
    return K.mean(gradient_penalty)


gradient_penalty = partial(gradient_penalty_loss, averaged_batch=averaged_batch,
                           penalty_weight=GRADIENT_PENALTY_WEIGHT)  # construct the gradient penalty
gradient_penalty.__name__ = 'gradient_penalty'

critic_training.compile(keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9, decay=0.0), loss=[
                        wasserstein_loss, wasserstein_loss, gradient_penalty])


positive_y = np.ones(BATCH_SIZE)
negative_y = -positive_y
# keras throws an error when calculating a loss without having a label -> needed for using the gradient penalty loss
dummy = np.zeros(BATCH_SIZE)


generator_loss = []
critic_loss = []

iterations_per_epoch = nsamples//(BATCH_SIZE*NCR)
#iterations_per_epoch=100
iters = 0
#use noiseReaderUproot when trying to save memory, also noiseReader places the trigger bin in the center of the trace
#filelist=['C:/users/Simon Hillmann/OneDrive/Dokumente/Uni/Bachelor/station_32_run_00225.root']
#reader=noiseReaderUproot.ARIANNADataReader(filelist)
#start=np.arange(400)
#np.random.shuffle(start)

for epoch in range(EPOCHS):
    print("epoch: ", epoch)
    np.random.shuffle(data)
    for iteration in range(iterations_per_epoch):
        #itera=iters %400
       # bunches=reader.get_events(bunchsize=BATCH_SIZE/4,entry_start=start[itera]*BATCH_SIZE/4,entry_stop=start[itera]*BATCH_SIZE/4+NCR*BATCH_SIZE/4,randomize=True)
        for  j in range(NCR):
            bunch=data[BATCH_SIZE*(j+iteration):BATCH_SIZE*(j++iteration+1)]
           # bunch=np.concatenate([bunch[i,:,:] for i in range(len(bunch[:,0,0]))])
            bunch = np.apply_along_axis(fft.time2freq, 1, bunch, 1*units.GHz)
            bunch[:,0]=0
            bunch=np.apply_along_axis(fft.freq2time,1,bunch,1*units.GHz)
            utils.make_trainable(g, False)  # freeze the generator during the critic training
            utils.make_trainable(critic, True)
            # generate noise batch for generator
            noise_batch = np.random.randn(BATCH_SIZE, latent_size)
            # take batch of shower maps
            
            critic_loss.append(critic_training.train_on_batch([noise_batch, bunch], [
                               negative_y, positive_y, dummy]))  # train the critic
        utils.make_trainable(g, True)  # freeze the generator during the critic training
        utils.make_trainable(critic, False)
        noise_batch = np.random.randn(BATCH_SIZE, latent_size)  # generate noise batch for generator
        generator_loss.append(generator_training.train_on_batch(
            [noise_batch], [positive_y]))  # train the generator
        iters += 1

        generated_signal = g.predict_on_batch(np.random.randn(BATCH_SIZE, latent_size))

        if iters % 100 == 1:
            print("iteration", iters)
            print("critic loss:", critic_loss[-1])
            print("generator loss:", generator_loss[-1])
            '''
            save synthetic traces throughout the training loop to see how the generator improves with every iteration
            fig = utils.plot_signal(generated_signal,256)
            fig.suptitle("iteration %i" % iters)
           
            fig.savefig("fouriersigmoid/fake_noise_iteration_%.6i.png" % iters)
            plt.close()
            '''
critic_loss = np.array(critic_loss)
#plot loss curves
plt.subplots(1, figsize=(10, 5))
plt.plot(np.arange(len(critic_loss)), critic_loss[:,0], color='red', markersize=12, label=r'Total')
plt.plot(np.arange(len(critic_loss)),
         critic_loss[:,1] + critic_loss[:, 2], color='green', label=r'Wasserstein', linestyle='dashed')
plt.plot(np.arange(len(critic_loss)),
         critic_loss[:, 3], color='royalblue', markersize=12, label=r'GradientPenalty', linestyle='dashed')
plt.legend(loc='upper right')
plt.xlabel(r'Iterations')
plt.ylabel(r'Critic Loss')
plt.ylim(-6, 3)
plt.savefig("fourier2/critic_loss.png")
plt.close()

generator_loss = np.array(generator_loss)

plt.subplots(1, figsize=(10, 5))
plt.plot(np.arange(len(generator_loss)), generator_loss, color='red', markersize=12, label=r'Total')
plt.legend(loc='upper right')
plt.xlabel(r'Iterations')
plt.ylabel(r'Loss')
plt.savefig("fourier2/generator_loss.png")
#save the generated networks
g.save('fourier2/generator_model')
critic.save('fourier2/critic_model')