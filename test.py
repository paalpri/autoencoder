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






model = load_model("weights.best.hdf5", )
#encoder = Model(autoencoder.layers[0].input, autoencoder.layers[13].output)
# summarize layers
print(model.summary())
# plot graph



def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag):
        super().__init__() 
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        img = data.astronaut()
        # Do something to the image
        img = (255 * skimage.util.random_noise(img)).astype('uint8')

        image = make_image(img)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, epoch)
        writer.close()

        return

tbi_callback = TensorBoardImage('Image Example')