from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import sys
import random
from pprint import pprint

# The main part of the code is taken from: https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/



# Parameters
m = 20
n_z = 2  # Number of encoder outputs
n_epoch = 10
input_size = 10


def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.0)
    return mu + K.exp(log_sigma / 2) * eps


# Q(z|X) -- encoder
inputs = Input(shape=(input_size,))
h_q = Dense(5, activation='relu')(inputs)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

# Sample z ~ Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])

# P(X|z) -- decoder
decoder_hidden = Dense(5, activation='relu')
decoder_out = Dense(input_size, activation='sigmoid')

h_p = decoder_hidden(z)
outputs = decoder_out(h_p)

# Overall VAE model, for reconstruction and training
vae = Model(inputs, outputs, name='autoencoder')

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(inputs, mu, name='encoder')

# Generator model, generate new data given latent variable z

d_in = Input(shape=(n_z,))
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


#checkpoint
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint, tensorboard]


filename = sys.argv[1]
vae.summary()
# makes a txt document into a list of arrays, one array for each line
data = np.genfromtxt(filename, delimiter=" ", dtype=int)

''' For test purposes, generates a random test file with numbers
data = [(random.sample(range(1, 20), 10)) for k in range(100)]
with open('testfile.txt', 'w') as f:
    for item in data:
        f.write(" ".join(map(str, item)) + "\n")
'''
# Validate the training with 20% of the data, leave 80 % for training
#train_valid_split = int(len(data)*0.80)
# Split the questions and answers into training and validating data

vae.compile(optimizer='adam', loss=vae_loss)
vae.fit(data, data, verbose='1', batch_size=m, nb_epoch=n_epoch, validation_split=0.2, callbacks=callbacks_list)
