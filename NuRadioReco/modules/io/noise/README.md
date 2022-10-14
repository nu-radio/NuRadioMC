# Noise generation for neutrino simulations using Generative Adversarial Networks 

This repository contains all the code for the Noise generation for neutrino simulations using Generative Adversarial Networks project. Report available here: [report](https://github.com/nu-radio/NuRadioMC)


## Installation

```bash
$ git clone https://github.com/nu-radio/NuRadioMC/tree/feature/noise_gan/NuRadioReco/modules/io/noise
$ cd noise/
$ sudo pip3 install -r requirements.txt
```

## Usage

### Data

An small subset of the actual raw and preprocessed data is available in the data folder.
There is not enough to train the network but it at least gives an good overview on how
the data looks like.

### Models

Several models are available in the models folder. 
Example on how to the models is in the notebooks folder

### Implementation

Code the implementation of the GANs is in wgan.py, generator.py and discriminator.py files which are stored in the models folder.

### Train
Example on how to train an model from scrath is in the notebooks folder.



