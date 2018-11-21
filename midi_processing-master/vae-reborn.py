import numpy as np
import tensorflow as tf
import sys

Input = tf.keras.layers.Input
Conv1D = tf.keras.layers.Conv1D
Dense = tf.keras.layers.Dense
#Conv1DTranspose = tf.keras.layers.Conv1DTranspose
Lambda = tf.keras.layers.Lambda

Model = tf.keras.models.Model
K = tf.keras.backend

INPUT_DIM = 10

INTERMEDIATE_DIMS = [16,32,64,128,256,128,64,32,16]

Z_DIM = 4

BATCH_SIZE = 32
EPOCHS = 5000

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
            line[i] = line[i]*(note_max - note_min) + note_min 
    return in_data

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(Z_DIM,), mean=0., stddev=.1)
    return mu + K.exp(0.5*log_sigma) * eps 

# Preprocess data
filename = sys.argv[1]
# makes a txt document into a list of arrays, one array for each line
data = np.genfromtxt(filename, delimiter=" ", dtype=float)
if len(data) % BATCH_SIZE != 0:
    data = data[:- (len(data) % BATCH_SIZE)]
data, note_min, note_max = minmax_norm(data)

vae_input = Input(shape=(INPUT_DIM,), name='encoder_input')
vae_f1 = Dense(INTERMEDIATE_DIMS[0], activation='relu')(vae_input)
vae_f2 = Dense(INTERMEDIATE_DIMS[1], activation='relu')(vae_f1)
vae_f3 = Dense(INTERMEDIATE_DIMS[2], activation='relu')(vae_f2)
vae_f4 = Dense(INTERMEDIATE_DIMS[3], activation='relu')(vae_f3)
vae_f5 = Dense(INTERMEDIATE_DIMS[4], activation='relu')(vae_f4)
vae_f6 = Dense(INTERMEDIATE_DIMS[5], activation='relu')(vae_f5)
vae_f7 = Dense(INTERMEDIATE_DIMS[6], activation='relu')(vae_f6)
vae_f8 = Dense(INTERMEDIATE_DIMS[7], activation='relu')(vae_f7)
#vae_z_mean = Dense(Z_DIM, name='z_mean')(vae_f4)
#vae_z_log_var = Dense(Z_DIM, name='z_log_var')(vae_f4)
#z = Lambda(sample_z, output_shape=(Z_DIM,), name='z')([vae_z_mean, vae_z_log_var])

vae_z = Dense(Z_DIM, activation='relu')(vae_f8)

#encoder = Model(vae_input, vae_z, name='encoder')
#encoder.summary()


#vae_z_input = Input(shape=(Z_DIM,))
vae_d_f1 = Dense(INTERMEDIATE_DIMS[-1], activation='relu')(vae_z)
vae_d_f2 = Dense(INTERMEDIATE_DIMS[-2], activation='relu')(vae_d_f1)
vae_d_f3 = Dense(INTERMEDIATE_DIMS[-3], activation='relu')(vae_d_f2)
vae_d_f4 = Dense(INTERMEDIATE_DIMS[-4], activation='relu')(vae_d_f3)
vae_d_f5 = Dense(INTERMEDIATE_DIMS[-5], activation='relu')(vae_d_f4)
vae_d_f6 = Dense(INTERMEDIATE_DIMS[-6], activation='relu')(vae_d_f5)
vae_d_f7 = Dense(INTERMEDIATE_DIMS[-7], activation='relu')(vae_d_f6)
vae_d_f8 = Dense(INTERMEDIATE_DIMS[-8], activation='relu')(vae_d_f7)
vae_output = Dense(INPUT_DIM, activation='sigmoid')(vae_d_f8)

#decoder = Model(vae_z_input,vae_output, name='decoder')
#decoder.summary()

#outputs = decoder(encoder(vae_input)) #?
vae = Model(vae_input, vae_output, name='vae')

def vae_r_loss(y_true, y_pred):
            #   return K.mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
            return 10 * K.mean(K.square(y_true - y_pred), axis = -1)

def vae_kl_loss(y_true, y_pred):
            return - 0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis = -1)

def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)

vae.compile(optimizer='adam', loss="mean_squared_error", metrics=['mse'])

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
callbacks_list = [earlystop]

vae.fit(data, data,
                shuffle=True,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=0.2,
                callbacks=callbacks_list)

vae.save_weights('./vae_reborn_weights.h5')

pred = vae.predict(data[:BATCH_SIZE], verbose=1, batch_size=BATCH_SIZE)
data = minmax_reverse(data, note_min, note_max)
pred = minmax_reverse(pred, note_min, note_max)

for i in range(3):
    print("Input: %s" %(' '.join(['%.4f'%j for j in data[i]])))
    print("Predi: %s" %(' '.join(['%.4f'%j for j in pred[i]])))