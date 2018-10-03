from __future__ import division, print_function, absolute_import
import keras
import glob
#import os
#import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data
from music21 import converter, instrument, note, chord
from music21 import *
import pretty_midi
from pprint import pprint
SONG_LENGTH = 128



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=16, dim=(128,128), n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size)) 

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x = self.__data_generation(list_IDs_temp)

        return x, x

    def on_epoch_end(self):
        'Updates indexes after each epoch' # if shuffle is true, makes the indexes random so we dont get the same input next time
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization


        x = np.zeros(self.batch_size, *self.dim, self.n_channels)
        SONG_LENGTH = 256
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            if i == SONG_LENGTH:
                continue
            # make the data from the file.
            roll = [[0] * SONG_LENGTH for x in range(128)]
            midi = converter.parse('data_2/' + ID + '.mdi') 
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                # Below ############------- Find out what recurse means, do we need this ? ---- ############
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    midi_numb = int(element.pitch.ps)
                    roll[midi_numb][counter] = 1
                elif isinstance(element, chord.Chord):
                    for p in element.pitches: 
                        midi_numb = int(p.ps) 
                        roll[midi_numb][counter] = 1

            # Store the input in the batch variable
            x[i,] = roll
        return x


