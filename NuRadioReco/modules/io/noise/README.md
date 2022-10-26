# Noise generation for neutrino simulations using Generative Adversarial Networks 

This repository contains all the code for the Noise generation for neutrino simulations using Generative Adversarial Networks project. The projected which was done as an Erasmus Internship between 29/8-28/10/22 by Daniel Hjelm. 
<!-- Report available here: [report](https://github.com/nu-radio/NuRadioMC) -->


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

Several models are available in the models folder. The model called 2048 is generating traces with 2048 samples. However, this model was not investigated in detail or optimized which is reflected in the performance.

How to generate noise with the models is displayed in generateNoise.ipynb in the same folder.

Example on how to load, train and analyze the performance of a model is shown in the train.ipynb notebook in the same folder.


### Implementation

The code for the implementation of the GANs is in wgan.py, generator.py and discriminator.py.

## Contact
Created by [Daniel Hjelm](mailto:dnl1@live.se) - feel free to contact me!



