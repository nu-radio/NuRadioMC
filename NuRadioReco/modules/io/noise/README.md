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

### How to generate data using a generator
```python

import numpy as np
from tensorflow import keras

# Load data
dataset = np.load('data/data_example.npy')

# Load generator
generator = keras.models.load_model('models/best/') 

# Decide how many events to generate
number_of_events = 1

# Generate signals using generator
noise = np.random.randn(number_of_events, 128)
noise = np.expand_dims(noise, axis=-1) 
generated_traces = generator.predict_on_batch(noise)
generated_traces = generated_traces[:,:,0]

# Scaling
generated_traces = generated_traces * dataset.std()+dataset.mean()
```
This is also displayed in generateNoise.ipynb in the models folder.
### Data

An small subset of the actual raw and preprocessed data is available in the data folder.
There is not enough to train the network but it at least gives an good overview on how
the data looks like.

The code for how the data is created and preprocessed is in the files named create_data.py and data_preprocessing.py respectively.

### Models

Several models are available in the models folder. The model called 2048 is generating traces with 2048 samples. However, this model was not investigated in detail or optimized which is reflected in the performance.

Example on how to load, train and analyze the performance of a model is shown in the train.ipynb notebook in the same folder.


### WGAN implementation

The code for the implementation of the GANs is in wgan.py, generator.py and discriminator.py.

### Metrics

The code for the statistical and visual metrics are in analyze.py.
The code for the astroparticle tests are in the folders named template and threshold.

## Contact
Created by [Daniel Hjelm](mailto:dnl1@live.se) - feel free to contact me!



