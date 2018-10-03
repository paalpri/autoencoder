from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

import matplotlib.pyplot as plt

from keras.datasets import mnist
import numpy as np

from keras.callbacks import TensorBoard

from data_generator import DataGenerator

BATCH_SIZE = 32

def decode(decoded):
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def model(input_song):

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_song)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_song, decoded)

    return autoencoder

def main(self):
    #Build model
    input_song = Input(shape=(128, 128, 1))  # adapt this if using `channels_first` image data format
    
    autoencoder = model(input_song)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    
    n_samples = 82
    params = {'dim': (128,128),
          'batch_size': 4,
          'n_channels': 1,
          'shuffle': True}
    partition = {'train': [str(i) for i in range(n_samples)],
             'validation': [str(i) for i in range(n_samples, n_samples+10)]}

    print(partition['train'])
    training_generator = DataGenerator(**params, list_IDs=partition['train'])
    validation_generator = DataGenerator(**params, list_IDs=partition['validation'])
    
    autoencoder.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=False,
                            workers=1)

    #Decode
    decoded = autoencoder.predict(x_test)
    decode(decoded)








    '''
    #Training data
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.    
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
    '''

    '''
    #Start training 
    autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    '''

   

if __name__ == '__main__':
    main(None)
