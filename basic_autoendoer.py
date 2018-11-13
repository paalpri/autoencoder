from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense, Lambda, Dropout, Flatten, Reshape
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

# The main part of the code is taken from: https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/


def minmax_norm(in_data):
    in_data = np.array(in_data, dtype=float)
    note_min = np.min(in_data)
    note_max = np.max(in_data)

    for line in in_data:
        for i in range(len(line)):
            #minmax scaling (0-1)
            line[i] = (line[i]- note_min)/(note_max-note_min)
    return in_data, note_min, note_max


def minmax_reverse(in_data, note_min, note_max):
    in_data = np.array(in_data, dtype=float)

    for line in in_data: 
        for i in range(len(line)):
            #Reverse minmax scaling
            line[i] = int(line[i]*(note_max - note_min) + note_min) 
    return in_data


# Delete the previous logs, for better tensorboard looks
if os.path.isdir("logs"):
    shutil.rmtree("logs")
# Parameters
batch_size = 10
n_epoch = 7
intermediate_dim = 32*12
latent_dim = 32*12

# Preprocess data
filename = sys.argv[1]
# makes a txt document into a list of arrays, one array for each line
data = np.genfromtxt(filename, delimiter=" ", dtype=int)
data = data[:- (len(data) % batch_size)]
#data, note_min, note_max = minmax_norm(data)
data_one = to_categorical(data)

original_dim1, original_dim2 = np.shape(data_one[0])
org = int(original_dim1*original_dim2)


def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(latent_dim,), mean=0., stddev=1.)
    return mu + K.exp(0.5*log_sigma) * eps 


# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=(original_dim1, original_dim2, ), name='encoder_input')
x = Flatten()(inputs)
x = Dense(intermediate_dim, activation='relu')(x)
x = Dense(intermediate_dim, activation='relu')(x)
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
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
x = Dense(intermediate_dim, activation='relu')(x)
x = Dense(org, activation='softmax')(x)
outputs = Reshape((original_dim1, original_dim2))(x)
# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    #recon = K.mean(K.square(y_pred - y_true), axis=-1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    #kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    #kl = - 0.5 * K.mean(1 + log_sigma - K.square(mu) - K.exp(log_sigma), axis=-1)
    #kl = - 0.5 * K.sum(1 + log_sigma - K.square(mu) - K.exp(log_sigma), axis=1)

    recon = K.sum(losses.binary_crossentropy(y_true, y_pred))
    kl = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(recon + kl)

Adam = optimizers.Adam(lr=0.001)
vae.compile(optimizer='adam', loss=vae_loss, metrics=['acc'])
# checkpoint
tensorboard = TensorBoard(log_dir="logs", write_graph=True, histogram_freq=0,)
checkpoint = ModelCheckpoint('training_weights.hdf5', monitor='val_loss', save_best_only=True, mode='min')
callbacks_list = [tensorboard]


vae.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])
vae.fit(data_one, data_one, verbose=1, shuffle=True, batch_size=batch_size, epochs=n_epoch, validation_split=0.2, callbacks=callbacks_list)

pred = vae.predict(data_one[:batch_size], verbose=1, batch_size=batch_size)
#data = minmax_reverse(data, note_min, note_max)
#pred = minmax_reverse(pred, note_min, note_max)

res = []
print(np.shape(pred[0]))
for i in (pred):
    song = []
    for x in i:
        song.append(np.argmax(x))
    res.append(song)

for i in range(3):
    print("Input: %s" %(data[i]))
    print("Predicted: %s" %(res[i]))

decoder.save('decoder_model.h5')

