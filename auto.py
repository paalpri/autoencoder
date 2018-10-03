
from __future__ import division, print_function, absolute_import
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
SONG_LENGTH = 256


def midi_to_roll(files):
    for filename, ids in enumerate(files): #(data_3/*.mdi)
        roll = [[0] * SONG_LENGTH for x in range(128)]
        counter = 0
        midi = converter.parse(filename) 
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts: # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if(counter < SONG_LENGTH): 
                print(element)
            if isinstance(element, note.Note):
                midi_numb = int(element.pitch.ps)
                if(counter < SONG_LENGTH):
                    roll[midi_numb][counter] = 1
            elif isinstance(element, chord.Chord):
                for p in element.pitches: 
                    midi_numb = int(p.ps) 
                    if(counter < SONG_LENGTH):  
                        roll[midi_numb][counter] = 1
        counter += 1
        yield roll
    





def main(_):
    files = glob.glob('data_3/*.mdi')
    outputs = midi_to_roll(files)
    pprint(outputs)


if __name__ == '__main__':
    main(None)