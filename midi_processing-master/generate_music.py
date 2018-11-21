import tensorflow as tf
import sys
import numpy as np
from create_midi import reconstruct_original, vae_midi
from my_parser_utils import pitch_to_note_conv, instrument_name
from perlin_noise2 import get_perlin_noise

Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Lambda = tf.keras.layers.Lambda
Flatten = tf.keras.layers.Flatten
Reshape = tf.keras.layers.Reshape

to_categorical = tf.keras.utils.to_categorical
losses = tf.keras.losses

Model = tf.keras.models.Model
K = tf.keras.backend


BATCH_SIZE = 16
EPOCHS = 100



original_midi = sys.argv[1]
id_track = int(sys.argv[2])
window = int(sys.argv[3])
shift = int(sys.argv[4])
is_d_major = bool(sys.argv[5])
original_dim1, original_dim2 = int(sys.argv[6]), int(sys.argv[7])
file_weights = sys.argv[8]


BATCH_SIZE = 16
EPOCHS = 100


# # read dataset
# filename = 'G_major_w16_id42.txt'
# #makes a txt document into a list of arrays, one array for each line
# original_data = np.genfromtxt(filename, delimiter=" ", dtype=int)
# if len(original_data) % BATCH_SIZE != 0:
#    original_data = original_data[:- (len(original_data) % BATCH_SIZE)]
# #data, note_min, note_max = minmax_norm(data)
# data = to_categorical(original_data)
# original_dim1, original_dim2 = np.shape(data[0])


Z_DIM = int(0.2*original_dim1*original_dim2)

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(Z_DIM,), mean=0., stddev=.1)
    return mu + K.exp(0.5*log_sigma) * eps 

def vae_loss(y_true, y_pred):    
    y_pred = K.clip(y_pred,1e-05,50)

    recon = K.sum(losses.categorical_crossentropy(y_true, y_pred))

    kl = - 0.5 * K.sum(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis=-1)

    return K.mean(recon + kl)

K.set_epsilon(1e-05)

INTERMEDIATE_DIMS = [int(original_dim1*original_dim2*0.5), int(original_dim1*original_dim2*0.5)]

vae_input = Input(shape=(original_dim1, original_dim2, ), name='encoder_input')
flatten_input = Flatten()(vae_input)
vae_f1 = Dense(INTERMEDIATE_DIMS[0], name='dense1', activation='relu')(flatten_input)
vae_f2 = Dense(INTERMEDIATE_DIMS[1], name='dense2', activation='relu')(vae_f1)
#vae_f3 = Dense(INTERMEDIATE_DIMS[2], activation='softplus')(vae_f2)
#vae_f4 = Dense(INTERMEDIATE_DIMS[3], activation='softplus')(vae_f3)
#vae_f5 = Dense(INTERMEDIATE_DIMS[4], activation='softplus')(vae_f4)
#vae_f6 = Dense(INTERMEDIATE_DIMS[5], activation='softplus')(vae_f5)
#vae_f7 = Dense(INTERMEDIATE_DIMS[6], activation='softplus')(vae_f6)
#vae_f8 = Dense(INTERMEDIATE_DIMS[7], activation='softplus')(vae_f7)
vae_z_mean = Dense(Z_DIM, name='z_mean')(vae_f2)
vae_z_log_var = Dense(Z_DIM, name='z_log_var')(vae_f2)
vae_z = Lambda(sample_z, output_shape=(Z_DIM,), name='z')([vae_z_mean, vae_z_log_var])

#vae_z = Dense(Z_DIM, activation='relu')(vae_f8)

encoder = Model(vae_input, vae_z, name='encoder')
#encoder.summary()


vae_z_input = Input(shape=(Z_DIM,),name='z_sampling')
vae_d_f1 = Dense(INTERMEDIATE_DIMS[-1], name='dense3', activation='relu')(vae_z_input)
vae_d_f2 = Dense(INTERMEDIATE_DIMS[-2], name='dense4', activation='relu')(vae_d_f1)
#vae_d_f3 = Dense(INTERMEDIATE_DIMS[-3], activation='softplus')(vae_d_f2)
#vae_d_f4 = Dense(INTERMEDIATE_DIMS[-4], activation='softplus')(vae_d_f3)
#vae_d_f5 = Dense(INTERMEDIATE_DIMS[-5], activation='softplus')(vae_d_f4)
#vae_d_f6 = Dense(INTERMEDIATE_DIMS[-6], activation='softplus')(vae_d_f5)
#vae_d_f7 = Dense(INTERMEDIATE_DIMS[-7], activation='softplus')(vae_d_f6)
#vae_d_f8 = Dense(INTERMEDIATE_DIMS[-8], activation='softplus')(vae_d_f7)
vae_final = Dense(original_dim1 * original_dim2, name='final', activation='softmax')(vae_d_f2)
vae_output = Reshape((original_dim1, original_dim2))(vae_final)

decoder = Model(vae_z_input,vae_output, name='decoder')
#decoder.summary()

outputs = decoder(encoder(vae_input)) #?
vae = Model(vae_input, outputs, name='vae')

vae.load_weights(file_weights)
encoder.load_weights(file_weights, by_name=True)
decoder.load_weights(file_weights, by_name=True)

#enc = encoder.predict(data[:BATCH_SIZE], verbose=1, batch_size=BATCH_SIZE)

#dec = decoder.predict(enc, verbose=1, batch_size=BATCH_SIZE)

#pred = vae.predict(data[:BATCH_SIZE], verbose=1, batch_size=BATCH_SIZE)



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

    new_song = []
    for s in song:
        for j in s:
            new_song.append(j)

    for note in all_notes:
        new_note = note
       
        new_note.pitch_number = new_song[i]
        new_note.id = note.octave*12 + new_song[i]-3
        if new_note.pitch_number<3:
            new_note.id += 12
        new_note.pitch  = pitch_to_note_conv[new_note.pitch_number]
        i+=1

        new_notes.append(new_note)

    vae_midi(new_notes, all_end_notes, elements, midiname)

music_c = to_categorical(music,num_classes=12)

song = []
song10 = []
song25 = []
song50 = []
noises10 = get_perlin_noise(len(music_c), Z_DIM, 0.1) 
noises25 = get_perlin_noise(len(music_c), Z_DIM, 0.25) 
noises50 = get_perlin_noise(len(music_c), Z_DIM, 0.5) 

for i in range(len(music_c)):
    enc = encoder.predict(music_c[i:i+1], verbose=1, batch_size=1)

    enc10 = enc.copy()
    enc25 = enc.copy()
    enc50 = enc.copy()
    for j in range(Z_DIM):
        if j <= 37:
            enc10[0][j] = 10
            enc25[0][j] = 10
            enc50[0][j] = 10
        else:
            enc10[0][j] += noises10[j][i]
            enc25[0][j] += noises25[j][i]
            enc50[0][j] += noises50[j][i]
     

    print(enc10)
    print(enc25)
    print(enc50)

    pred = decoder.predict(enc, verbose=1, batch_size=1)
    pred10 = decoder.predict(enc10, verbose=1, batch_size=1)
    pred25 = decoder.predict(enc25, verbose=1, batch_size=1)
    pred50 = decoder.predict(enc50, verbose=1, batch_size=1)

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