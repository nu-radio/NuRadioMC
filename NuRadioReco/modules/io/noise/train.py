import numpy as np
import matplotlib.pyplot as plt
from wgan import WGAN
from NuRadioReco.utilities import fft
from NuRadioReco.utilities import units
from NuRadioReco.framework import base_trace
import tensorflow.keras.backend as K

def train():

    # Load data
    data = np.load('../data_preprocessed.npy')

    # Create WGAN
    wgan = WGAN()
    
    # Keras throws an error when calculating a loss without having a label -> needed for using the gradient penalty loss
    positive_y = np.ones(wgan.batch_size)
    negative_y = -positive_y
    dummy = np.zeros(wgan.batch_size) 

    # Create arrays for generator and critic loss
    generator_loss = []
    critic_loss = []

    # Training parameters
    EPOCHS = 50
    nsamples = len(data)
    critic_iterations = 5
    iterations_per_epoch = nsamples*2//(wgan.batch_size*critic_iterations)
    iters = 0
    print(iterations_per_epoch)
    

    
    for epoch in range(EPOCHS):
        
        print("Epoch: ", epoch)

        for iteration in range(iterations_per_epoch):
        
            for j in range(critic_iterations):
                
                # Pick data in batches 
                bunch=data[wgan.batch_size*(j+iteration):wgan.batch_size*(j++iteration+1)]
                bunch = np.expand_dims(bunch, axis=-1)  

                # Generate noise
                noise_batch = np.random.randn(wgan.batch_size, wgan.latent_size)
                noise_batch = np.expand_dims(noise_batch, axis=-1) 

                
                # Train critic
                critic_loss.append(wgan.critic_training.train_on_batch([noise_batch, bunch], [negative_y, positive_y, dummy]))
            

            # Generate noise batch for generator
            noise_batch = np.random.randn(wgan.batch_size, wgan.latent_size)
            noise_batch = np.expand_dims(noise_batch, axis=-1) 
            
            # Train the generator
            generator_loss.append(wgan.generator_training.train_on_batch([noise_batch], [positive_y]))  
            iters+=1
            
            # Printing losses and plotting example traces
            if iters % 300 == 1:
                print("Iteration", iters)
                print("Critic loss:", critic_loss[-1])
                print("Generator loss:", generator_loss[-1])
                
                # Generate signals
                noise = np.random.randn(wgan.batch_size, wgan.latent_size)
                noise = np.expand_dims(noise, axis=-1) 
                generated_signals = wgan.generator.predict_on_batch(noise)
                generated_signals = generated_signals[:,:,0]
                
                
                # Plot data
                
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.set_size_inches(18.5, 10.5, forward=True)
                ax1.title.set_text('Time domain')
                ax2.title.set_text('Frequency domain')
                
                

                ax1.plot(data[0], label = "Example trace")
                for i in range(3):
                    ax1.plot(generated_signals[i], alpha=0.5)
                
                
                # Plot frequency
                ax2.plot(abs(fft.time2freq(data[0], 3.2*units.GHz)), label = "Example trace")
                for i in range(3):
                    ax2.plot(abs(fft.time2freq(generated_signals[i], 3.2*units.GHz)),alpha=0.2)
                    
                ax1.legend()
                ax2.legend()
                plt.show()



if __name__ == '__main__':
    train()