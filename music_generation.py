from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense, Lambda, Dropout, Flatten, Reshape
from keras.activations import softmax
from keras.utils import to_categorical
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
import numpy as np
import keras.backend as K
from keras import losses, optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
import tensorflow as tf
import sys
import random
from pprint import pprint
import shutil, os
import pickle

# The main part of the code is taken from: https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

# Delete the previous logs, for better tensorboard looks
if os.path.isdir("logs"):
    shutil.rmtree("logs")

# Preprocess data
filename = sys.argv[1]
# makes a txt document into a list of arrays, one array for each line
data = np.genfromtxt(filename, delimiter=" ", dtype=int)
data = data[:- (len(data) % int(sys.argv[2]))]
data_one = to_categorical(data) #One-hot encoding

print(np.shape(data_one))

# Parameters
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
learning_rate = float(sys.argv[4])
latent_scale = float(sys.argv[5])
original_dim1, original_dim2 = np.shape(data_one[0])
org = int(original_dim1*original_dim2)
intermediate_dim = int(org*0.5)
latent_dim = int(org*latent_scale)
K.set_epsilon(1e-05)


def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(latent_dim,), mean=0., stddev=.1)
    return mu + K.exp(0.5*log_sigma) * eps 


def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    #recon = K.mean(K.square(y_pred - y_true), axis=-1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    #kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    #kl = - 0.5 * K.mean(1 + log_sigma - K.square(mu) - K.exp(log_sigma), axis=-1)
    #kl = - 0.5 * K.sum(1 + log_sigma - K.square(mu) - K.exp(log_sigma), axis=1 )

    y_pred = K.clip(y_pred,1e-05,50)

    recon = K.sum(losses.categorical_crossentropy(y_true, y_pred))

    kl = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    return K.mean(recon + kl)


# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=(original_dim1, original_dim2, ), name='encoder_input')
x = Flatten()(inputs)
x = Dense(intermediate_dim, activation='softplus')(x)
#x = Dropout(0.2)(x)
x = Dense(intermediate_dim, activation='softplus')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='softplus')(latent_inputs)
#x = Dropout(0.2)(x)
x = Dense(intermediate_dim, activation='softplus')(x)
x = Dense(org, activation='softmax')(x)
outputs = Reshape((original_dim1, original_dim2))(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

encoder = load_model('saved_models/encoder_model.h5', {'latent_dim':latent_dim, 'vae_loss':vae_loss}, compile=False)
decoder = load_model('saved_models/decoder_model.h5', {'latent_dim':latent_dim, 'vae_loss':vae_loss}, compile=False)

latent_var = encoder.predict(data_one[:1], batch_size=1)

print(np.shape(latent_var))
print((latent_var[2]))

tweeked = np.copy(latent_var[2])

tweeked[0][0] = tweeked[0][0]*100
print(tweeked)

reproduction = decoder.predict(latent_var[2], batch_size=1)

generation = decoder.predict(tweeked, batch_size=1)


res_r = []
res_g = []
for i,j in zip(reproduction,generation):
    song_r = []
    song_g = []
    for x,y in zip(i,j):
        song_r.append(np.argmax(x))
        song_g.append(np.argmax(y))
    res_r.append(song_r)
    res_g.append(song_g)
print(data[0])
print(res_r)
print(res_g)