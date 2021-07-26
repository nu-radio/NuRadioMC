# Arianna Noise GAN
This project is designed to train a Wasserstein-Genrative-Adversarial-Network (WGAN) to simulate realistic radio noise traces for the radio neutrino detector ARIANNA.  Working with this project requires the tensorflow package, the uproot package and the awkward package to be installed. 
The project contains two files:
* utils&#46;py
* arianna_noise_gan&#46;py  

## utils&#46;py
This file defines some utilitary functions used in the main code.  
In this file one can also access and modifie the network architectures used in the training loop.  
## arianna_noise_gan&#46;py
This file contains the main training loop for the WGAN.  
Here one can change the training parameters and perform the network training.  
The training loop requires data, which needs to be read in from a &#46;root file.


