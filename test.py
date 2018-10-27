from keras.models import load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from keras.callbacks import TensorBoard
from data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint


input_layer = Input(shape=(16, 16, 8)) 

decoder0 = Conv2D(8, 3, activation='relu', padding='same')(input_layer)
x = UpSampling2D((2, 2))(decoder0)
decoder1 = Conv2D(8, 3, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(decoder1)
decoder2 = Conv2D(8, 3, activation='relu', padding = 'same')(x)
x = UpSampling2D((2, 2))(decoder2)
decoded = Conv2D(1, 1, activation='sigmoid', padding='same')(x)

autoencoder = Model(input_layer, decoded)

autoencoder.load_weights("weights.best.hdf5",by_name = True)
print(autoencoder.summary())


#autoencoder.load_weights(filepath)
#checkpoint
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', )
    callbacks_list = [checkpoint]#, [TensorBoard(log_dir='/tmp/autoencoder')]]

    



model = load_model("weights.best.hdf5", )
#encoder = Model(autoencoder.layers[0].input, autoencoder.layers[13].output)
# summarize layers
print(model.summary())
# plot graph