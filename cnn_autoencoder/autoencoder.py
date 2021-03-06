from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from keras.callbacks import TensorBoard
from data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf

BATCH_SIZE = 32

def decode(decoded):
    print("Is there a one ? ")
    decoded = K.round(decoded)
    decoded = K.eval(decoded)
    for i in range(decoded.shape[0]):
        for j in range(decoded.shape[1]):
            for k in range(decoded.shape[2]):
                if(decoded[i][j][k] == 1):
                    print(1)
    
    return
def model(input_song):

    x = Conv2D(16, 3, activation='relu', padding='same')(input_song)
    x = Dropout(0.5)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, 3, activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, 3, activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (?, 16,16, 8) i.e. 128-dimensional
    

    x = Conv2D(8, 3, activation='relu', padding='same')(encoded)
    x = Dropout(0.5)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, 3, activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, 3, activation='relu', padding = 'same')(x)
    x = Dropout(0.5)(x)
    x = UpSampling2D((2, 2))(x)
    print(np.shape(x))
    decoded = Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_song, decoded)

    return autoencoder

def main(filepath):
    #Build model
    input_song = Input(shape=(128, 128, 1))  # adapt this if using `channels_first` image data format
    
    autoencoder = model(input_song)
    # If we want to load from earlier run
    #autoencoder.load_weights(filepath)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',  metrics=['accuracy'])

    
    n_samples = 81
    train_folder = 'training_data'
    validation_folder = 'validation'
    test_folder = 'test_data'

    fold = os.listdir(train_folder) # dir is your directory path
    nmb_train = len(fold)
    nmb_validation =int(nmb_train * 0.3)

    params = {'dim': (128,128),
          'batch_size': 5,
          'n_channels': 1,
          'shuffle': True}
    partition = {'train': [str(i) for i in range(nmb_train - nmb_validation )],
             'validation': [str(i) for i in range(nmb_train - nmb_validation, nmb_train)],
             'test' : [str(i) for i in range(n_samples+10, n_samples+11)]}

    training_generator = DataGenerator(**params, filepath = train_folder, list_IDs=partition['train'])
    validation_generator = DataGenerator(**params,filepath = train_folder, list_IDs=partition['validation'])
    #test_generator = DataGenerator(**params,list_IDs=partition['test'])

    
    #checkpoint
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)
    callbacks_list = [checkpoint, tensorboard]

    
   
    autoencoder.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            epochs = 10,
                            callbacks = callbacks_list)

    #Decode
    decoded = autoencoder.predict_generator(generator = validation_generator)
    decode(decoded)


if __name__ == '__main__':
    main("weights.best.hdf5")
