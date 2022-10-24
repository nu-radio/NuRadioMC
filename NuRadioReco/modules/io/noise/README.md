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


### Implementation

The code for the implementation of the GANs is in wgan.py, generator.py and discriminator.py.

### Example
Example on how to train load, train and analyze the performance of a model is shown in the example.ipynb notebook in the notebooks folder



