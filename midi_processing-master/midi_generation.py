import tensorflow as tf
import sys
import numpy as np
from create_midi import reconstruct_original, vae_midi
from my_parser_utils import pitch_to_note_conv, instrument_name

Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Lambda = tf.keras.layers.Lambda
Flatten = tf.keras.layers.Flatten
Reshape = tf.keras.layers.Reshape

to_categorical = tf.keras.utils.to_categorical
losses = tf.keras.losses

Model = tf.keras.models.Model
K = tf.keras.backend

load_model = tf.keras.models.load_model


BATCH_SIZE = 32
EPOCHS = 5000

Z_DIM = 3


original_midi = sys.argv[5]
id_track = int(sys.argv[6])
window = int(sys.argv[7])
shift = int(sys.argv[8])
is_d_major = bool(sys.argv[9])
original_dim1, original_dim2 = int(sys.argv[10]), int(sys.argv[11])

file_weights = sys.argv[12]

# Parameters
batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
learning_rate = float(sys.argv[3])
latent_scale = float(sys.argv[4])
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
x = Dense(intermediate_dim, activation='relu')(x)
#x = Dropout(0.2)(x)
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
#x = Dropout(0.2)(x)
x = Dense(intermediate_dim, activation='relu')(x)
x = Dense(org, activation='softmax')(x)
outputs = Reshape((original_dim1, original_dim2))(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

# # checkpoint
# tensorboard = TensorBoard(log_dir="logs", write_graph=True, histogram_freq=0,)
# checkpoint = ModelCheckpoint('training_weights.hdf5', monitor='val_loss', save_best_only=True, mode='min')
# callbacks_list = [tensorboard]

# Adam = optimizers.Adam(lr=learning_rate)
# vae.compile(optimizer=Adam, loss=vae_loss, metrics=['acc'])
# encoder.compile(optimizer=Adam, loss=vae_loss, metrics=['acc'])
# decoder.compile(optimizer=Adam, loss=vae_loss, metrics=['acc'])
# #Start training
# history = vae.fit(data_one, data_one, verbose=1, shuffle=True, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=callbacks_list)

# pred = vae.predict(data_one[:batch_size], verbose=1, batch_size=batch_size)


vae.load_weights(file_weights)
encoder.load_weights(file_weights, by_name=True)
decoder.load_weights(file_weights, by_name=True)

# tweeked = np.copy(latent_var[2])

# tweeked[0][0] = tweeked[0][0]*100
# print(tweeked)

# reproduction = decoder.predict(latent_var[2], batch_size=1)

# generation = decoder.predict(tweeked, batch_size=1)



aux = {5:0, 7:1, 9:2, 10:3, 0:4, 2:5, 4:6}
aux_back = {0:5, 1:7, 2:9, 3:10, 4:0, 5:2, 6:4}

all_notes, all_end_notes, elements = reconstruct_original(original_midi, id_track,window)
music = []
for i in range(0, len(all_notes), window):
    inp = []
    if is_d_major:
        inp+=[aux[all_notes[i+j].pitch_number] for j in range(window)]
    else: 
        inp+=[all_notes[i+j].pitch_number for j in range(window)]
    music += [inp]


def convert_pred(pred):
    for i in pred:
        s = []
        for x in i:
            s.append(np.argmax(x))
    return s

def midi_create(song, midiname):
    new_notes = []
    i = 0
    j = 0
    for note in all_notes:
        new_note = note
        if i <= len(all_notes)-window:
            if is_d_major:
                new_note.pitch_number = aux_back[song[i][0]]
                new_note.id = note.octave*12 + aux_back[song[i][0]]-3
            else:
                new_note.pitch_number = song[i][0]
                new_note.id = note.octave*12 + song[i][0]-3
            if new_note.pitch_number<3:
                new_note.id += 12
            new_note.pitch  = pitch_to_note_conv[new_note.pitch_number]
            i+=1
        else:
            if is_d_major:
                new_note.pitch_number = aux_back[song[-1][j+1]]
                new_note.id = note.octave*12 + aux_back[song[-1][j+1]]-3
            else:
                new_note.pitch_number = song[-1][j+1]
                new_note.id = note.octave*12 + song[-1][j+1]-3
            if new_note.pitch_number<3:
                new_note.id += 12
            new_note.pitch  = pitch_to_note_conv[new_note.pitch_number]
            j+=1
        new_notes.append(new_note)

    vae_midi(new_notes, all_end_notes, elements, midiname)

music_c = to_categorical(music,num_classes=7)

song = []
song10 = []
song25 = []
song50 = []
noises10 = get_perlin_noise(len(music_c), Z_DIM, 0.1) 
noises25 = get_perlin_noise(len(music_c), Z_DIM, 0.25) 
noises50 = get_perlin_noise(len(music_c), Z_DIM, 0.5) 

for i in range(len(music_c)):
    enc = encoder.predict(music_c[i:i+1], verbose=1, batch_size=1)

    enc10 = enc
    enc25 = enc
    enc50 = enc
    for j in range(Z_DIM):
        enc10[0][j] += noises10[i][j]
        enc25[0][j] += noises25[i][j]
        enc50[0][j] += noises50[i][j]

    pred = decoder.predict(enc[2], verbose=1, batch_size=1)
    pred10 = decoder.predict(enc10[2], verbose=1, batch_size=1)
    pred25 = decoder.predict(enc25[2], verbose=1, batch_size=1)
    pred50 = decoder.predict(enc50[2], verbose=1, batch_size=1)

    song.append(convert_pred(pred))
    song10.append(convert_pred(pred10))
    song25.append(convert_pred(pred25))
    song50.append(convert_pred(pred50))

midi_create(song, 'reconstructed.mid')
midi_create(song10, 'noise10.mid')
midi_create(song25, 'noise25.mid')
midi_create(song50, 'noise50.mid')

#res = []
# for i in (dec):
#     song = []
#     for x in i:
#         song.append(np.argmax(x))
#     res.append(song)

# res2 = []
# for i in (pred):
#     song = []
#     for x in i:
#         song.append(np.argmax(x))
#     res2.append(song)

# for i in range(3):
#     print("Input: %s" %(' '.join(['%.4f'%j for j in original_data[i]])))
#     print("Enccc: %s" %(' '.join(['%.4f'%j for j in enc[i]])))
#     print("Deccc: %s" %(' '.join(['%.4f'%j for j in res[i]])))
#     print("Predi: %s" %(' '.join(['%.4f'%j for j in res2[i]])))