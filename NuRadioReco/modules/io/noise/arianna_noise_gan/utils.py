import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from NuRadioReco.modules.io.noise import noiseImporterUproot as ni
layers = keras.layers


def ReadInData(filename:str):
    '''Reads in the trainings data'''
    
    data = ni.noiseImporter()
    data.begin([filename])
    return data.nevts,data.data




def plot_signal(dat,length):
    '''plots 4 traces with length length'''
    fig, ax= plt.subplots()
    num=np.arange(length)
    for i in range(4):
        
        ax.plot(num,dat[i,:],'-',label='Channel'+str(i))
        ax.set_xlabel('time bin')
        ax.set_ylabel('E')
    ax.legend()
    return fig

def generator_model(latent_size):
    """ Generator network """
    '''modify network as you want'''
    latent = layers.Input(shape=(latent_size,), name="noise")
    z = layers.Dense((latent_size),activation='relu')(latent)
    
    z=layers.Dense(256)(z) #output layer has to have the same shape as a noise trace 
    return keras.models.Model(latent, z, name="generator")


# build critic
# Feel free to modify the critic model
def critic_model(latent_size):
    image = layers.Input(shape=(256), name="images")
    x=layers.Dense(latent_size,activation='relu')(image)
   
    x = layers.Dense(1,activation='sigmoid')(x) 
    # no activation!
    return keras.models.Model(image, x, name="critic")


def make_trainable(model, trainable):
    ''' Freezes/unfreezes the weights in the given model '''
    for layer in model.layers:
        # print(type(layer))
        if type(layer) is layers.BatchNormalization:
            layer.trainable = True
        else:
            layer.trainable = trainable
 #random weighted average           
class UniformLineSampler(tf.keras.layers.Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        weights = K.random_uniform((self.batch_size, 1))
        return(weights * inputs[0]) + ((1 - weights) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]
