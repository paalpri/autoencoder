from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
import numpy as np
import keras.backend as K
import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
import tensorflow as tf
import sys
import random
from pprint import pprint
import shutil, os
'''
RANDOMMMMMMMM
ødmømddøm
'''

# The main part of the code is taken from: https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
keras.__version__
tf.__version__
# Dele the previous logs, for better tensorboard looks
if os.path.isdir("logs"):
    print("What")
    shutil.rmtree("logs")
# Parameters
batch_size = 64
n_epoch = 20
original_dim = 15
intermediate_dim = 5
latent_dim = 2

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
    return mu + K.exp(log_sigma / 2) * eps # /2 på log sigma


# Q(z|X) -- encoder
inputs = Input(shape=(original_dim,))
h_q = Dense(intermediate_dim, activation='relu')(inputs)
mu = Dense(latent_dim, activation='linear')(h_q)
log_sigma = Dense(latent_dim, activation='linear')(h_q)

# Sample z ~ Q(z|X)
z = Lambda(sample_z, output_shape=(latent_dim,))([mu, log_sigma])

# P(X|z) -- decoder
decoder_hidden = Dense(intermediate_dim, activation='relu')
decoder_out = Dense(original_dim, activation='sigmoid')
h_p = decoder_hidden(z)
outputs = decoder_out(h_p)

# Overall VAE model, for reconstruction and training
vae = Model(inputs, outputs, name='autoencoder')

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(inputs, mu, name='encoder')

# Generator model, generate new data given latent variable z
d_in = Input(shape=(latent_dim,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out, name='decoder')


def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl


vae.compile(optimizer='adam', loss=vae_loss, metrics=['acc'])
vae.summary()
# checkpoint
tensorboard = TensorBoard(log_dir="logs", write_graph=True, histogram_freq=0,)
checkpoint = ModelCheckpoint('training_weights.hdf5', monitor='val_loss', save_best_only=True, mode='min')
callbacks_list = [tensorboard]

filename = sys.argv[1]

# makes a txt document into a list of arrays, one array for each line
data = np.genfromtxt(filename, delimiter=" ", dtype=int)
data = np.array(data, dtype=float)
note_min = np.min(data)
note_max = np.max(data)

for line in data:
    for i in range(len(line)):
        #minmax scaling (0-1)
        line[i] = (line[i]- note_min)/(note_max-note_min)
        #Reverse minmax scaling
        #i = int(i*(note_max - note_min) + note_min) 
        

vae.compile(optimizer='adam', loss=vae_loss, metrics=['accuracy'])
vae.fit(data, data, verbose=1, batch_size=batch_size, epochs=n_epoch, validation_split=0.2, callbacks=callbacks_list)
decoder.save('decoder_model.h5')

