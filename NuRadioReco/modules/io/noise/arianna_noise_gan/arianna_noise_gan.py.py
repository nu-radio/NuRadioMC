#!/usr/bin/env pygpu
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import utils
tf.compat.v1.disable_eager_execution()  # gp loss won't work with eager
layers = keras.layers


# basic training parameter
EPOCHS = 10
GRADIENT_PENALTY_WEIGHT = 10
BATCH_SIZE = 100
NCR = 5
latent_size = 256


# load trainings data
N,data,trig = utils.ReadInData('C:/users/Simon Hillmann/OneDrive/Dokumente/Uni/Bachelor/station_32_run_00225.root')
nsamples = N

pulse=np.sinc(np.arange(256)-100)
data=np.concatenate([data[i,:,:] for i in range(len(data[:,0,0]))])
print(np.shape(data))


#data=np.random.normal(size=(N,256))
# plot real signal patterns
fig = utils.plot_signal(data,length=256)
fig.savefig("./random_signal.png")


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

iterations_per_epoch = nsamples // (NCR * BATCH_SIZE)
#iterations_per_epoch=100
iters = 0

for epoch in range(EPOCHS):
    print("epoch: ", epoch)

    for iteration in range(iterations_per_epoch):

        for j in range(NCR):
            utils.make_trainable(g, False)  # freeze the generator during the critic training
            utils.make_trainable(critic, True)
            # generate noise batch for generator
            noise_batch = np.random.randn(BATCH_SIZE, latent_size)
            # take batch of shower maps
            shower_batch = data[BATCH_SIZE*(j+iteration):BATCH_SIZE*(j++iteration+1)]
            critic_loss.append(critic_training.train_on_batch([noise_batch, shower_batch], [
                               negative_y, positive_y, dummy]))  # train the critic
        utils.make_trainable(g, True)  # freeze the generator during the critic training
        utils.make_trainable(critic, False)
        noise_batch = np.random.randn(BATCH_SIZE, latent_size)  # generate noise batch for generator
        generator_loss.append(generator_training.train_on_batch(
            [noise_batch], [positive_y]))  # train the generator
        iters += 1

        generated_maps = g.predict_on_batch(np.random.randn(BATCH_SIZE, latent_size))

        if iters % 300 == 1:
            print("iteration", iters)
            print("critic loss:", critic_loss[-1])
            print("generator loss:", generator_loss[-1])

            fig = utils.plot_signal(generated_maps,256)
            fig.suptitle("iteration %i" % iters)
           # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig("./fake_noise_iteration_%.6i.png" % iters)


critic_loss = np.array(critic_loss)

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
plt.savefig("./critic_loss.png")


generator_loss = np.array(generator_loss)

plt.subplots(1, figsize=(10, 5))
plt.plot(np.arange(len(generator_loss)), generator_loss, color='red', markersize=12, label=r'Total')
plt.legend(loc='upper right')
plt.xlabel(r'Iterations')
plt.ylabel(r'Loss')
plt.savefig("./generator_loss.png")
