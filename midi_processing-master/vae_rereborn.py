import numpy as np
import tensorflow as tf
import sys

Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Lambda = tf.keras.layers.Lambda
Flatten = tf.keras.layers.Flatten
Reshape = tf.keras.layers.Reshape

to_categorical = tf.keras.utils.to_categorical
losses = tf.keras.losses
optimizers = tf.keras.optimizers

Model = tf.keras.models.Model
K = tf.keras.backend

BATCH_SIZE = 16
EPOCHS = 100


# read dataset
filename = '/home/johannes/github/autoencoder/processed_datasets/G_major_id42_s1_pres.txt'
#makes a txt document into a list of arrays, one array for each line
original_data = np.genfromtxt(filename, delimiter=" ", dtype=int)
if len(original_data) % BATCH_SIZE != 0:
   original_data = original_data[:- (len(original_data) % BATCH_SIZE)]
#data, note_min, note_max = minmax_norm(data)
data = to_categorical(original_data)
original_dim1, original_dim2 = np.shape(data[0])


Z_DIM = int(0.2*original_dim1*original_dim2)

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(Z_DIM,), mean=0., stddev=.1)
    return mu + K.exp(0.5*log_sigma) * eps 



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

def vae_loss(y_true, y_pred):    
    y_pred = K.clip(y_pred,1e-05,50)

    recon = K.sum(losses.categorical_crossentropy(y_true, y_pred))

    kl = - 0.5 * K.sum(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis=-1)

    return K.mean(recon + kl)

Adam = optimizers.Adam(lr=0.001)
vae.compile(optimizer=Adam, loss=vae_loss, metrics=['acc'])

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
    print("Input: %s" %(' '.join(['%.4f'%j for j in original_data[i]])))
    print("Predi: %s" %(' '.join(['%.4f'%j for j in res[i]])))

encoder.save('encoder_model.h5')
decoder.save('decoder_model.h5')
vae.save('vae_model.h5')


#from keras.models import load_model
#model = load_model('model.h5')